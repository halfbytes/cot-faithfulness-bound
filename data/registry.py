"""Benchmark lookup, generation caps and data preparation."""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List
from experiments.config import get
from data.schema import Benchmark, Instance, write_jsonl
from data.cryptobench import CryptoBench, complexity_report, exemplars_from_generator, generate_cryptobench
from data.benchmarks import CLASSES, EXPECTED_EVAL_SIZES, PREPARERS, VALIDATION_N, public_benchmark

ALL_BENCHMARKS = ['gsm8k', 'arc_challenge', 'math_algebra', 'strategyqa', 'boardgameqa', 'gpqa_diamond', 'cryptobench']


def benchmark_dir(cfg: Dict[str, Any], name: str) -> str:
    override = get(cfg, f'data.benchmarks.{name}.path')
    if override:
        return override
    return os.path.join(get(cfg, 'data.root', 'data'), name)


def get_benchmark(cfg: Dict[str, Any], name: str) -> Benchmark:
    d = benchmark_dir(cfg, name)
    if name == 'cryptobench':
        ex_seed = int(get(cfg, 'data.benchmarks.cryptobench.exemplar_seed', 12345))
        short = bool(get(cfg, 'data.benchmarks.cryptobench.short_exemplars', True))
        return CryptoBench(d, exemplars_from_generator(ex_seed, short=short))
    if name in CLASSES:
        return public_benchmark(name, d)
    raise KeyError(f'unknown benchmark {name!r}; known: {ALL_BENCHMARKS}')


def generation_cap(cfg: dict, name: str) -> int:
    caps = get(cfg, 'generation.per_benchmark_T', {}) or {}
    if name in caps:
        return int(caps[name])
    return int(get(cfg, 'generation.max_new_tokens_default', 512))


def make_cryptobench(cfg: Dict[str, Any], out_dir: str | None = None) -> Dict[str, Any]:
    """Write the evaluation, training and validation splits and their statistics."""
    sub = get(cfg, 'data.benchmarks.cryptobench', {}) or {}
    out_dir = out_dir or benchmark_dir(cfg, 'cryptobench')
    sizes = sub.get('sizes', {'cipher': 200, 'modular': 200, 'hash': 100})
    kw = dict(n_cipher=int(sizes['cipher']), n_modular=int(sizes['modular']), n_hash=int(sizes['hash']))
    eval_items = generate_cryptobench(seed=int(sub.get('seed', 0)), **kw)
    n_train = int(sub.get('n_train', 2000))
    frac = {k: v / sum(kw.values()) for k, v in kw.items()}
    train_items = generate_cryptobench(seed=int(sub.get('train_seed', 1)), n_cipher=round(n_train * frac['n_cipher']),
                                       n_modular=round(n_train * frac['n_modular']), n_hash=round(n_train * frac['n_hash']))
    n_val = int(sub.get('n_validation', VALIDATION_N))
    n_val_c, n_val_m = (round(n_val * frac['n_cipher']), round(n_val * frac['n_modular']))
    val_items = generate_cryptobench(seed=int(sub.get('validation_seed', 2)), n_cipher=n_val_c, n_modular=n_val_m,
                                     n_hash=n_val - n_val_c - n_val_m)
    for split, items in (('eval', eval_items), ('train', train_items), ('validation', val_items)):
        _tag(items, split)
        write_jsonl(os.path.join(out_dir, f'{split}.jsonl'), items)
    stats = {'seeds': {'eval': int(sub.get('seed', 0)), 'train': int(sub.get('train_seed', 1)), 'validation': int(sub.get('validation_seed', 2))},
             'sizes': {'eval': len(eval_items), 'train': len(train_items), 'validation': len(val_items)},
             'complexity_eval': complexity_report(eval_items, lz_method=str(get(cfg, 'complexity.lz_method', 'lz78')))}
    with open(os.path.join(out_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    return stats


def _tag(items: List[Instance], split: str) -> None:
    for it in items:
        it.id = f'{it.id}-{split}'
        it.meta['split'] = split


def prepare_public(cfg: Dict[str, Any], names: List[str], token: str | None = None) -> Dict[str, Dict[str, int]]:
    seed = int(get(cfg, 'data.split_seed', 0))
    out: Dict[str, Dict[str, int]] = {}
    for name in names:
        if name == 'cryptobench':
            out[name] = make_cryptobench(cfg)['sizes']
            continue
        prep = PREPARERS[name]
        kwargs = {}
        if name == 'boardgameqa':
            kwargs['subset'] = str(get(cfg, 'data.benchmarks.boardgameqa.subset', 'BoardgameQA-Main-depth2'))
        if name == 'strategyqa':
            kwargs['carve_validation'] = bool(get(cfg, 'data.benchmarks.strategyqa.carve_validation', False))
        counts = prep(benchmark_dir(cfg, name), seed, token, **kwargs)
        exp = EXPECTED_EVAL_SIZES.get(name)
        if name == 'strategyqa' and kwargs['carve_validation']:
            exp = exp - VALIDATION_N
        if exp is not None and counts['eval'] != exp:
            n_eval = counts['eval']
            print(f'[warn] {name}: eval split has {n_eval} items, expected {exp}')
        out[name] = counts
    return out
