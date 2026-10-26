"""Chain-of-thought generation over a benchmark split with per-token and effective-capacity statistics."""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from theory import capacity as theory
from experiments.config import get
from data.schema import Benchmark, Instance
from data.prompts import build_prompt
from data.registry import generation_cap, get_benchmark
from experiments.model import GenOutput, encode_prompt, generate, load_model_and_tokenizer, model_dims, split_trace_and_answer
from experiments.seeding import seed_everything


def generation_dir(cfg: Dict[str, Any], model_key: str, bench: str, split: str, cap: int) -> str:
    k = int(get(cfg, 'prompting.k_shot', 8))
    return os.path.join(get(cfg, 'paths.results', 'results'), 'generation', model_key, bench, f'{split}_T{cap}_k{k}')


def prompt_for(cfg: Dict[str, Any], benchmark: Benchmark, inst: Instance, k_shot: Optional[int] = None) -> str:
    k = int(get(cfg, 'prompting.k_shot', 8)) if k_shot is None else k_shot
    return build_prompt(inst.question, benchmark.exemplars(k), delimiter=str(get(cfg, 'prompting.delimiter', 'The answer is')))


def effective_entropy(records: Sequence[Dict[str, Any]], n_traces: int, mode: str = 'nll', aggregation: str = 'pooled') -> float:
    """Mean per-token surprisal ('nll') or next-token entropy ('shannon') over the first n_traces records."""
    key = {'nll': 'logprob_bits', 'shannon': 'entropy_bits'}[mode]
    recs = list(records)[:n_traces]
    if aggregation == 'pooled':
        vals: List[float] = []
        for r in recs:
            vals.extend([-x for x in r[key]] if mode == 'nll' else r[key])
        return float(np.mean(vals)) if vals else 0.0
    means = [float(np.mean([-x for x in r[key]] if mode == 'nll' else r[key])) for r in recs if r[key]]
    return float(np.mean(means)) if means else 0.0


def run_generation(cfg: Dict[str, Any], model_key: str, bench_name: str, split: str = 'eval', cap: Optional[int] = None,
                   limit: Optional[int] = None, model_and_tok=None, k_shot: Optional[int] = None,
                   out_dir: Optional[str] = None, bottleneck=None, tag: str = '') -> Dict[str, Any]:
    """Generate a trace for every instance of the split; write traces, prompt-final states and statistics."""
    seed_everything(int(get(cfg, 'seed', 0)))
    model_cfg = cfg['models'][model_key]
    benchmark = get_benchmark(cfg, bench_name)
    cap = int(cap or generation_cap(cfg, bench_name))
    out_dir = out_dir or generation_dir(cfg, model_key, bench_name, split + tag, cap)
    os.makedirs(out_dir, exist_ok=True)
    limit = limit if limit is not None else get(cfg, 'generation.limit')
    instances = benchmark.load(split, limit=limit)
    model, tokenizer = model_and_tok or load_model_and_tokenizer(cfg, model_cfg)
    dims = model_dims(model, tokenizer)
    h0_layer = int(get(cfg, 'generation.h0_layer', -1))
    stop_strings = list(get(cfg, 'prompting.stop_strings', ['\nQ:']))
    delimiter = str(get(cfg, 'prompting.delimiter', 'The answer is'))
    records: List[Dict[str, Any]] = []
    h0s: List[np.ndarray] = []
    collect = split == 'validation' and str(get(cfg, 'representations.h0_pca_fit', 'validation_positions')) == 'validation_positions'
    all_states: List[np.ndarray] = []
    n_correct = 0
    for i, inst in enumerate(instances):
        prompt = prompt_for(cfg, benchmark, inst, k_shot=k_shot)
        ids = encode_prompt(tokenizer, prompt)
        if bottleneck is not None:
            bottleneck.set_generation_mode()
        out: GenOutput = generate(model, tokenizer, ids, cap, h0_layer=h0_layer, stop_strings=stop_strings,
                                  collect_states=collect)
        if out.states is not None:
            all_states.append(out.states)
        trace_text, pred = split_trace_and_answer(out.text, delimiter)
        correct = benchmark.is_correct(pred, inst) if pred != '' else False
        n_correct += int(correct)
        answer_tokens = len(tokenizer.encode(pred, add_special_tokens=False)) if pred else 0
        records.append({'id': inst.id, 'prompt_len': out.prompt_len, 'token_ids': out.token_ids, 'text': out.text,
                        'trace_text': trace_text, 'answer_pred': pred, 'answer_gold': inst.answer, 'correct': bool(correct),
                        'n_tokens': len(out.token_ids), 'answer_tokens': answer_tokens, 'finished': out.finished,
                        'logprob_bits': [round(x, 5) for x in out.logprob_bits],
                        'entropy_bits': [round(x, 5) for x in out.entropy_bits]})
        h0s.append(out.h0.astype(np.float32))
        if (i + 1) % max(1, -(-len(instances) // 10)) == 0 or i + 1 == len(instances):
            print(f'[generate] {model_key}/{bench_name}/{split}{tag} {i + 1}/{len(instances)} acc={n_correct / (i + 1):.3f}')
    with open(os.path.join(out_dir, 'traces.jsonl'), 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    np.save(os.path.join(out_dir, 'h0.npy'), np.stack(h0s) if h0s else np.zeros((0, dims['d']), np.float32))
    if collect and all_states:
        np.save(os.path.join(out_dir, 'states.npy'), np.concatenate(all_states).astype(np.float16))
    by_cat: Dict[str, Dict[str, Any]] = {}
    cat_of = {i.id: i.meta.get('category') for i in instances}
    for r in records:
        cat = cat_of.get(r['id'])
        if cat is None:
            continue
        d = by_cat.setdefault(cat, {'n': 0, 'n_correct': 0, 'trace_tokens': [], 'answer_tokens': []})
        d['n'] += 1
        d['n_correct'] += int(r['correct'])
        d['trace_tokens'].append(r['n_tokens'])
        d['answer_tokens'].append(r['answer_tokens'])
    per_category = {c: {'n': d['n'], 'n_correct': d['n_correct'], 'accuracy': d['n_correct'] / max(1, d['n']),
                        'mean_trace_tokens': float(np.mean(d['trace_tokens'])),
                        'mean_answer_tokens': float(np.mean(d['answer_tokens']))} for c, d in by_cat.items()}
    n_traces = int(get(cfg, 'entropy.n_traces', 1000))
    mode = str(get(cfg, 'entropy.mode', 'nll'))
    agg = str(get(cfg, 'entropy.aggregation', 'pooled'))
    H_hat = effective_entropy(records, n_traces, mode=mode, aggregation=agg)
    lens = [r['n_tokens'] for r in records]
    alens = [r['answer_tokens'] for r in records]
    stats = {'model': model_key, 'benchmark': bench_name, 'split': split + tag, 'N': len(records), 'T_cap': cap, 'dims': dims,
             'accuracy': n_correct / max(1, len(records)), 'n_correct': n_correct,
             'mean_trace_tokens': float(np.mean(lens)) if lens else 0.0, 'std_trace_tokens': float(np.std(lens)) if lens else 0.0,
             'mean_answer_tokens': float(np.mean(alens)) if alens else 0.0, 'std_answer_tokens': float(np.std(alens)) if alens else 0.0,
             'frac_reached_cap': float(np.mean([r['finished'] == 'cap' for r in records])) if records else 0.0,
             'frac_with_answer': float(np.mean([r['answer_pred'] != '' for r in records])) if records else 0.0,
             'per_category': per_category, 'H_hat_bits_per_token': H_hat, 'H_hat_mode': mode,
             'H_hat_nll': effective_entropy(records, n_traces, mode='nll', aggregation=agg),
             'H_hat_shannon': effective_entropy(records, n_traces, mode='shannon', aggregation=agg),
             'C_eff': theory.effective_capacity(cap, H_hat), 'C_TV_ceil': theory.capacity_bits_ceil(cap, dims['v']),
             'C_TV': theory.capacity_bits(cap, dims['v']),
             'above_chance_p': theory.binomial_tail(n_correct, len(records), benchmark.chance_accuracy)
             if benchmark.chance_accuracy > 0 and records else None}
    with open(os.path.join(out_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    return stats


def load_generation(out_dir: str):
    """(records, prompt-final state array) written by run_generation."""
    records = []
    with open(os.path.join(out_dir, 'traces.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    h0 = np.load(os.path.join(out_dir, 'h0.npy'))
    return (records, h0)


def load_states(out_dir: str):
    """Per-position state array written by run_generation for the validation split, or None."""
    p = os.path.join(out_dir, 'states.npy')
    return np.load(p) if os.path.exists(p) else None
