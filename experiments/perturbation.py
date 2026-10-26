"""Perturbation sensitivity: resample one step of a trace and regenerate the remainder."""
from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from theory import capacity as theory
from experiments.config import get
from data.registry import generation_cap, get_benchmark
from experiments.generation import generation_dir, load_generation, prompt_for
from experiments.model import TokenPolicy, encode_prompt, generate, load_model_and_tokenizer, split_trace_and_answer
from experiments.seeding import seed_everything

SENTENCE_END = re.compile('[.!?\\n]')


def step_boundaries(text: str, mode: str = 'sentence') -> List[int]:
    """Character offsets at which each step ends."""
    if mode == 'line':
        bounds = [m.end() for m in re.finditer('\\n+', text)]
    else:
        bounds = [m.end() for m in SENTENCE_END.finditer(text)]
    bounds = [b for b in bounds if b < len(text)]
    bounds.append(len(text))
    return bounds


def step_spans(text: str, mode: str = 'sentence') -> List[tuple]:
    """(start, end) character spans of the non-empty steps of text."""
    spans, prev = ([], 0)
    for b in step_boundaries(text, mode):
        if b > prev and text[prev:b].strip():
            spans.append((prev, b))
        prev = b
    return spans


def candidate_steps(text: str, delimiter: str, mode: str = 'sentence', exclude_answer_step: bool = True) -> List[tuple]:
    """Spans of the steps eligible for resampling; with exclude_answer_step no part of the step holding the last delimiter is eligible."""
    if not exclude_answer_step:
        return step_spans(text, mode)
    trace_text, _ = split_trace_and_answer(text, delimiter)
    spans = step_spans(trace_text, mode)
    head = text[:text.rfind(delimiter)].rstrip(' \t') if delimiter in text else ''
    if spans and head and not (head[-1] == '\n' if mode == 'line' else SENTENCE_END.match(head[-1])):
        spans = spans[:-1]
    return spans


def token_prefix_len(tokenizer, token_ids: Sequence[int], char_offset: int) -> int:
    """Smallest number of leading tokens whose decoded text reaches char_offset characters."""
    lo, hi = (0, len(token_ids))
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(list(token_ids[:mid]), skip_special_tokens=True)) < char_offset:
            lo = mid + 1
        else:
            hi = mid
    return lo


def perturb_instance(cfg, model, tokenizer, benchmark, inst, record: Dict[str, Any], cap: int, rng: np.random.Generator,
                     torch_gen: torch.Generator) -> Optional[Dict[str, Any]]:
    delimiter = str(get(cfg, 'prompting.delimiter', 'The answer is'))
    mode = str(get(cfg, 'perturbation.step_delimiter', 'sentence'))
    text = record['text']
    spans = candidate_steps(text, delimiter, mode, bool(get(cfg, 'perturbation.exclude_answer_step', True)))
    if not spans:
        return None
    s_idx = int(rng.integers(0, len(spans)))
    start_char = spans[s_idx][0]
    token_ids = record['token_ids']
    prefix_len = token_prefix_len(tokenizer, token_ids, start_char)
    prompt = prompt_for(cfg, benchmark, inst)
    ids = encode_prompt(tokenizer, prompt)
    policy = TokenPolicy(forced_prefix=list(token_ids[:prefix_len]), sample_until_boundary=True,
                         temperature=float(get(cfg, 'perturbation.temperature', 1.0)),
                         max_sample_tokens=int(get(cfg, 'perturbation.max_step_tokens', 64)),
                         boundary_fn=lambda seg: bool(SENTENCE_END.search(seg)) if mode == 'sentence' else '\n' in seg,
                         generator=torch_gen)
    out = generate(model, tokenizer, ids, cap, h0_layer=int(get(cfg, 'generation.h0_layer', -1)),
                   stop_strings=list(get(cfg, 'prompting.stop_strings', ['\nQ:'])), policy=policy)
    _, new_answer = split_trace_and_answer(out.text, delimiter)
    changed = new_answer.strip() != record['answer_pred'].strip()
    text_changed = out.text.strip() != text.strip()
    return {'id': record['id'], 'step_index': s_idx, 'n_steps': len(spans), 'prefix_tokens': prefix_len,
            'n_sampled': out.n_sampled, 'answer_before': record['answer_pred'], 'answer_after': new_answer,
            'changed': bool(changed), 'text_changed': bool(text_changed),
            'correct_after': bool(benchmark.is_correct(new_answer, inst)) if new_answer else False}


def run_perturbation(cfg: Dict[str, Any], model_key: str, bench_name: str, cap: Optional[int] = None, model_and_tok=None,
                     limit: Optional[int] = None, gen_dir: Optional[str] = None, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Sensitivity over the perturbation seeds for one (model, benchmark)."""
    cap = int(cap or generation_cap(cfg, bench_name))
    benchmark = get_benchmark(cfg, bench_name)
    model, tokenizer = model_and_tok or load_model_and_tokenizer(cfg, cfg['models'][model_key])
    gen_dir = gen_dir or generation_dir(cfg, model_key, bench_name, 'eval', cap)
    records, _ = load_generation(gen_dir)
    instances = {i.id: i for i in benchmark.load('eval', limit=get(cfg, 'generation.limit'))}
    if limit:
        records = records[:limit]
    seeds = list(get(cfg, 'perturbation.seeds', [0, 1, 2, 3, 4]))
    per_seed: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    for s in seeds:
        seed_everything(int(s))
        rng = np.random.default_rng(s)
        tg = torch.Generator()
        tg.manual_seed(int(s))
        details = []
        for rec in records:
            inst = instances.get(rec['id'])
            if inst is None:
                continue
            d = perturb_instance(cfg, model, tokenizer, benchmark, inst, rec, cap, rng, tg)
            if d is not None:
                d['seed'] = int(s)
                details.append(d)
        ps = float(np.mean([d['changed'] for d in details])) if details else 0.0
        frac_text = float(np.mean([d['text_changed'] for d in details])) if details else 0.0
        per_seed.append({'seed': int(s), 'PS': ps, 'n': len(details), 'frac_trace_changed': frac_text})
        all_details.extend(details)
        print(f'[perturbation] {model_key}/{bench_name} seed={s} PS={ps:.3f} (n={len(details)})')
    vals = np.array([p['PS'] for p in per_seed], dtype=np.float64)
    out = {'model': model_key, 'benchmark': bench_name, 'T_cap': cap, 'PS_mean': float(vals.mean()) if len(vals) else 0.0,
           'PS_std': float(vals.std(ddof=0)) if len(vals) else 0.0, 'per_seed': per_seed, 'seeds': seeds,
           'answer_space_size': benchmark.answer_space_size,
           'PS_upper_bound': None if benchmark.answer_space_size is None else theory.ps_upper_bound(benchmark.answer_space_size),
           'n_instances': len(records)}
    out_dir = out_dir or os.path.join(get(cfg, 'paths.results', 'results'), 'perturbation', model_key, bench_name, f'T{cap}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'ps.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(out_dir, 'details.jsonl'), 'w', encoding='utf-8') as f:
        for d in all_details:
            f.write(json.dumps(d) + '\n')
    return out

