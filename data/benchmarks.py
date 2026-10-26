"""Download and conversion of the public benchmarks into the common schema."""
from __future__ import annotations
import io
import json
import os
import random
import re
import urllib.request
import zipfile
from typing import Callable, Dict, List, Optional
from data.schema import Benchmark, Instance, normalize_boardgame, normalize_choice, normalize_math, normalize_numeric, normalize_yes_no, strip_answer, write_jsonl
from data.prompts import EXEMPLARS

VALIDATION_N = 100
LETTERS = 'ABCD'
MC_LETTERS = 'ABCDEFGH'
BOARDGAMEQA_URL = 'https://storage.googleapis.com/gresearch/BoardgameQA/BoardgameQA.zip'


def _hf_load(dataset: str, config: Optional[str], split: str, token: Optional[str] = None):
    from datasets import load_dataset
    kwargs = {}
    if token:
        kwargs['token'] = token
    return load_dataset(dataset, config, split=split, **kwargs) if config else load_dataset(dataset, split=split, **kwargs)


def _split_off_validation(items: List[Instance], seed: int, n_val: int = VALIDATION_N):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    val_idx = set(idx[:n_val])
    val = [items[i] for i in sorted(val_idx)]
    rest = [items[i] for i in range(len(items)) if i not in val_idx]
    return (val, rest)


def _mc_question(stem: str, choices: List[str]) -> str:
    return stem.strip() + '\n' + '\n'.join(f'({MC_LETTERS[i]}) {c.strip()}' for i, c in enumerate(choices))


def _gsm8k_instance(row: Dict, i: int, split: str) -> Instance:
    sol = row['answer']
    gold = sol.split('####')[-1].strip().replace(',', '')
    rationale = sol.split('####')[0].strip()
    rationale = re.sub('<<[^>]*>>', '', rationale)
    return Instance(id=f'gsm8k-{split}-{i}', question=row['question'].strip(), answer=gold,
                    trace=f'{rationale} The answer is {gold}.', meta={'split': split})


def prepare_gsm8k(out_dir: str, seed: int, token: Optional[str] = None) -> Dict[str, int]:
    test = [_gsm8k_instance(r, i, 'test') for i, r in enumerate(_hf_load('openai/gsm8k', 'main', 'test', token))]
    train = [_gsm8k_instance(r, i, 'train') for i, r in enumerate(_hf_load('openai/gsm8k', 'main', 'train', token))]
    val, train_rest = _split_off_validation(train, seed)
    return _write_all(out_dir, test, train_rest, val)


class GSM8K(Benchmark):
    name = 'gsm8k'

    def normalize(self, raw: str) -> str:
        return normalize_numeric(raw)


def _boxed(solution: str) -> Optional[str]:
    m = re.search('\\\\boxed\\s*{', solution)
    if not m:
        return None
    depth, start = (0, m.end() - 1)
    for j in range(start, len(solution)):
        if solution[j] == '{':
            depth += 1
        elif solution[j] == '}':
            depth -= 1
            if depth == 0:
                return solution[start + 1:j]
    return None


def _math_instance(row: Dict, i: int, split: str) -> Instance:
    gold = _boxed(row['solution']) or ''
    sol = row['solution'].strip()
    return Instance(id=f'math-{split}-{i}', question=row['problem'].strip(), answer=gold, trace=f'{sol} The answer is {gold}.',
                    meta={'split': split, 'level': row.get('level'), 'type': row.get('type')})


def prepare_math_algebra(out_dir: str, seed: int, token: Optional[str] = None) -> Dict[str, int]:
    test = [_math_instance(r, i, 'test') for i, r in enumerate(_hf_load('EleutherAI/hendrycks_math', 'algebra', 'test', token))]
    train = [_math_instance(r, i, 'train') for i, r in enumerate(_hf_load('EleutherAI/hendrycks_math', 'algebra', 'train', token))]
    val, train_rest = _split_off_validation(train, seed)
    return _write_all(out_dir, test, train_rest, val)


class MathAlgebra(Benchmark):
    name = 'math_algebra'

    def normalize(self, raw: str) -> str:
        return normalize_math(strip_answer(raw))

    def is_correct(self, pred_raw: str, inst: Instance) -> bool:
        return self.normalize(pred_raw) == normalize_math(inst.answer)


def _arc_instance(row: Dict, i: int, split: str) -> Optional[Instance]:
    labels = list(row['choices']['label'])
    texts = list(row['choices']['text'])
    key = row['answerKey']
    if key not in labels or not 2 <= len(texts) <= len(MC_LETTERS):
        return None
    gold = MC_LETTERS[labels.index(key)]
    return Instance(id=f'arc-{split}-{i}', question=_mc_question(row['question'], texts), answer=gold, trace=None,
                    meta={'split': split, 'original_id': row.get('id'), 'n_choices': len(texts)})


def prepare_arc_challenge(out_dir: str, seed: int, token: Optional[str] = None) -> Dict[str, int]:
    def conv(split):
        out = []
        for i, r in enumerate(_hf_load('allenai/ai2_arc', 'ARC-Challenge', split, token)):
            inst = _arc_instance(r, i, split)
            if inst is not None:
                out.append(inst)
        return out
    test, train, val_src = (conv('test'), conv('train'), conv('validation'))
    return _write_all(out_dir, test, train, val_src[:VALIDATION_N])


class MultipleChoice(Benchmark):
    answer_space_size = 4
    chance_accuracy = 0.25

    def normalize(self, raw: str) -> str:
        return normalize_choice(raw, MC_LETTERS)


class ARCChallenge(MultipleChoice):
    name = 'arc_challenge'


def _strategyqa_instance(row: Dict, i: int) -> Instance:
    ans = 'yes' if str(row['answer']).strip().lower() in ('true', 'yes', '1') else 'no'
    facts = row.get('facts')
    if isinstance(facts, str):
        try:
            facts = json.loads(facts.replace("'", '"'))
        except Exception:
            facts = [facts]
    rationale = ' '.join(f.strip() for f in facts or [])
    qid = row.get('qid', i)
    return Instance(id=f'strategyqa-{qid}', question=row['question'].strip(), answer=ans,
                    trace=f'{rationale} The answer is {ans}.' if rationale else None, meta={'term': row.get('term')})


def prepare_strategyqa(out_dir: str, seed: int, token: Optional[str] = None, carve_validation: bool = False) -> Dict[str, int]:
    """Convert the answered StrategyQA questions (the original train split, served as the `test` split of `wics/strategy-qa`)."""
    rows = list(_hf_load('wics/strategy-qa', 'strategyQA', 'test', token))
    items = [_strategyqa_instance(r, i) for i, r in enumerate(rows)]
    val, rest = _split_off_validation(items, seed)
    return _write_all(out_dir, rest if carve_validation else items, rest, val)


class StrategyQA(Benchmark):
    name = 'strategyqa'
    answer_space_size = 2
    chance_accuracy = 0.5

    def normalize(self, raw: str) -> str:
        return normalize_yes_no(raw)


def _boardgame_instance(row: Dict, i: int, split: str) -> Instance:
    label = row['label'].strip().lower()
    proof = row['proof'].strip()
    return Instance(id=f'boardgameqa-{split}-{i}', question=row['example'].strip(), answer=label,
                    trace=f'{proof} The answer is {label}.', meta={'split': split, 'goal': row.get('goal')})


def prepare_boardgameqa(out_dir: str, seed: int, token: Optional[str] = None, subset: str = 'BoardgameQA-Main-depth2',
                        url: str = BOARDGAMEQA_URL) -> Dict[str, int]:
    cache = os.path.join(out_dir, 'BoardgameQA.zip')
    if not os.path.exists(cache):
        os.makedirs(out_dir, exist_ok=True)
        urllib.request.urlretrieve(url, cache)
    with zipfile.ZipFile(cache) as z:
        def load(name):
            with z.open(f'BoardgameQA/{subset}/{name}.json') as f:
                return json.load(io.TextIOWrapper(f, encoding='utf-8'))
        test, train, valid = (load('test'), load('train'), load('valid'))
    test_i = [_boardgame_instance(r, i, 'test') for i, r in enumerate(test)]
    train_i = [_boardgame_instance(r, i, 'train') for i, r in enumerate(train)]
    val_i = [_boardgame_instance(r, i, 'valid') for i, r in enumerate(valid)][:VALIDATION_N]
    return _write_all(out_dir, test_i, train_i, val_i)


class BoardgameQA(Benchmark):
    name = 'boardgameqa'
    answer_space_size = 3
    chance_accuracy = 1.0 / 3.0

    def normalize(self, raw: str) -> str:
        return normalize_boardgame(raw)


def _gpqa_instance(row: Dict, i: int, split: str, seed: int) -> Instance:
    options = [row['Correct Answer'], row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3']]
    rng = random.Random(f'{seed}-{split}-{i}')
    order = list(range(4))
    rng.shuffle(order)
    choices = [str(options[j]).strip() for j in order]
    gold = LETTERS[order.index(0)]
    expl = row.get('Explanation')
    return Instance(id=f'gpqa-{split}-{i}', question=_mc_question(row['Question'], choices), answer=gold,
                    trace=f'{str(expl).strip()} The answer is {gold}.' if expl else None,
                    meta={'split': split, 'subdomain': row.get('Subdomain'), 'record_id': row.get('Record ID')})


def prepare_gpqa_diamond(out_dir: str, seed: int, token: Optional[str] = None) -> Dict[str, int]:
    token = token or os.environ.get('HF_TOKEN')
    diamond = list(_hf_load('Idavidrein/gpqa', 'gpqa_diamond', 'train', token))
    main = list(_hf_load('Idavidrein/gpqa', 'gpqa_main', 'train', token))
    diamond_q = {r['Question'].strip() for r in diamond}
    extra = [r for r in main if r['Question'].strip() not in diamond_q]
    eval_i = [_gpqa_instance(r, i, 'diamond', seed) for i, r in enumerate(diamond)]
    extra_i = [_gpqa_instance(r, i, 'main', seed) for i, r in enumerate(extra)]
    val, rest = _split_off_validation(extra_i, seed)
    return _write_all(out_dir, eval_i, rest, val)


class GPQADiamond(MultipleChoice):
    name = 'gpqa_diamond'


def _write_all(out_dir: str, eval_items: List[Instance], train_items: List[Instance], val_items: List[Instance]) -> Dict[str, int]:
    return {'eval': write_jsonl(os.path.join(out_dir, 'eval.jsonl'), eval_items),
            'train': write_jsonl(os.path.join(out_dir, 'train.jsonl'), train_items),
            'validation': write_jsonl(os.path.join(out_dir, 'validation.jsonl'), val_items)}


PREPARERS: Dict[str, Callable[..., Dict[str, int]]] = {'gsm8k': prepare_gsm8k, 'math_algebra': prepare_math_algebra,
                                                       'arc_challenge': prepare_arc_challenge, 'strategyqa': prepare_strategyqa,
                                                       'boardgameqa': prepare_boardgameqa, 'gpqa_diamond': prepare_gpqa_diamond}
CLASSES: Dict[str, type] = {'gsm8k': GSM8K, 'math_algebra': MathAlgebra, 'arc_challenge': ARCChallenge, 'strategyqa': StrategyQA,
                            'boardgameqa': BoardgameQA, 'gpqa_diamond': GPQADiamond}
EXPECTED_EVAL_SIZES = {'gsm8k': 1319, 'math_algebra': 1187, 'arc_challenge': 1172, 'strategyqa': 2290, 'boardgameqa': 1000,
                       'gpqa_diamond': 198, 'cryptobench': 500}


def public_benchmark(name: str, data_dir: str) -> Benchmark:
    return CLASSES[name](data_dir, EXEMPLARS[name])
