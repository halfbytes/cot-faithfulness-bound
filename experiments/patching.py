"""Attention-recovery estimation by splicing the attention cache from a different input."""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from estimation import mine
from theory import capacity as theory
from experiments.config import get
from data.registry import generation_cap, get_benchmark
from experiments.generation import generation_dir, load_generation, prompt_for, run_generation
from experiments.model import encode_prompt, hidden_at_last_position, load_model_and_tokenizer, model_dims, resolve_device, sliding_window
from estimation.representations import PCAProjector, Standardizer, encode_texts
from experiments.seeding import seed_everything


@torch.no_grad()
def _kv_patched_hidden(model, prompt_ids: List[int], prefix_ids: List[int], donor_prompt_ids: List[int],
                       donor_prefix_ids: List[int], h0_layer: int, patch: bool) -> np.ndarray:
    """State that produces the next token, with the cached trace entries before the last prefix token taken from the donor."""
    from transformers import DynamicCache
    device = next(model.parameters()).device

    def run(ids):
        cache = DynamicCache()
        return model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), past_key_values=cache,
                     use_cache=True, output_hidden_states=True)
    n_prompt, n_prefix = (len(prompt_ids), len(prefix_ids))
    window = sliding_window(model)
    if patch and window and n_prompt + n_prefix > window:
        raise ValueError(f'the attention-cache splice needs the sequence within the {window}-token attention window; '
                         'use patching.mechanism=token_prefix')
    out = run(prompt_ids + prefix_ids)
    if not patch or n_prefix == 0:
        return out.hidden_states[h0_layer][0, -1].float().cpu().numpy()
    donor = run(donor_prompt_ids + donor_prefix_ids)
    cache, donor_cache = (out.past_key_values, donor.past_key_values)
    d_prompt = len(donor_prompt_ids)
    n = min(n_prefix, len(donor_prefix_ids))
    if hasattr(cache, 'key_cache'):
        for layer in range(len(cache.key_cache)):
            cache.key_cache[layer][:, :, n_prompt:n_prompt + n, :] = donor_cache.key_cache[layer][:, :, d_prompt:d_prompt + n, :]
            cache.value_cache[layer][:, :, n_prompt:n_prompt + n, :] = donor_cache.value_cache[layer][:, :, d_prompt:d_prompt + n, :]
    else:
        for layer in range(len(cache.layers)):
            cache.layers[layer].keys[:, :, n_prompt:n_prompt + n, :] = donor_cache.layers[layer].keys[:, :, d_prompt:d_prompt + n, :]
            cache.layers[layer].values[:, :, n_prompt:n_prompt + n, :] = donor_cache.layers[layer].values[:, :, d_prompt:d_prompt + n, :]
    cache.crop(n_prompt + n_prefix - 1)
    step = model(input_ids=torch.tensor([[prefix_ids[-1]]], dtype=torch.long, device=device), past_key_values=cache,
                 use_cache=True, output_hidden_states=True)
    return step.hidden_states[h0_layer][0, -1].float().cpu().numpy()


def _states_at_step(model, tokenizer, cfg, benchmark, instances, records, step: int, donor_shift: int = 1):
    """(clean, patched) state arrays at a 1-indexed generation step over the records long enough to reach it."""
    h0_layer = int(get(cfg, 'generation.h0_layer', -1))
    mechanism = str(get(cfg, 'patching.mechanism', 'kv_cache'))
    if mechanism not in ('kv_cache', 'token_prefix'):
        raise ValueError(f'unknown patching mechanism {mechanism!r}')
    if step < 2:
        return (np.zeros((0, 1), np.float32), np.zeros((0, 1), np.float32), [])
    clean, patched, keep = ([], [], [])
    usable = [r for r in records if len(r['token_ids']) >= step]
    for j, rec in enumerate(usable):
        inst = instances.get(rec['id'])
        if inst is None:
            continue
        prompt_ids = encode_prompt(tokenizer, prompt_for(cfg, benchmark, inst))
        prefix = list(rec['token_ids'][:step - 1])
        donor = usable[(j + donor_shift) % len(usable)]
        donor_inst = instances.get(donor['id'])
        donor_prefix = list(donor['token_ids'][:step - 1])
        if mechanism == 'kv_cache' and donor_inst is not None:
            donor_prompt_ids = encode_prompt(tokenizer, prompt_for(cfg, benchmark, donor_inst))
            clean.append(_kv_patched_hidden(model, prompt_ids, prefix, donor_prompt_ids, donor_prefix, h0_layer, patch=False))
            patched.append(_kv_patched_hidden(model, prompt_ids, prefix, donor_prompt_ids, donor_prefix, h0_layer, patch=True))
        else:
            clean.append(hidden_at_last_position(model, prompt_ids + prefix, h0_layer))
            patched.append(hidden_at_last_position(model, prompt_ids + donor_prefix[:-1] + prefix[-1:], h0_layer))
        keep.append(rec)
    if not clean:
        return (np.zeros((0, 1), np.float32), np.zeros((0, 1), np.float32), [])
    return (np.stack(clean).astype(np.float32), np.stack(patched).astype(np.float32), keep)


def run_patching(cfg: Dict[str, Any], model_key: str, bench_name: str, cap: Optional[int] = None, model_and_tok=None) -> Dict[str, Any]:
    """Recovery estimate at each configured step on the validation split."""
    seed_everything(int(get(cfg, 'seed', 0)))
    cap = int(cap or generation_cap(cfg, bench_name))
    benchmark = get_benchmark(cfg, bench_name)
    model, tokenizer = model_and_tok or load_model_and_tokenizer(cfg, cfg['models'][model_key])
    dims = model_dims(model, tokenizer)
    n_examples = int(get(cfg, 'patching.n_examples', 100))
    gen_dir = generation_dir(cfg, model_key, bench_name, 'validation', cap)
    if not os.path.exists(os.path.join(gen_dir, 'traces.jsonl')):
        run_generation(cfg, model_key, bench_name, 'validation', cap=cap, model_and_tok=(model, tokenizer), limit=n_examples)
    records, _ = load_generation(gen_dir)
    records = records[:n_examples]
    instances = {i.id: i for i in benchmark.load('validation', limit=n_examples)}
    answer_source = str(get(cfg, 'representations.answer_source', 'model'))
    mine_kw = dict(iterations=int(get(cfg, 'patching.iterations', 1000)), lr=float(get(cfg, 'mine.lr', 0.0001)),
                   hidden_dims=tuple(get(cfg, 'mine.hidden_dims', [512, 256, 128])), proj_dim=int(get(cfg, 'mine.proj_dim', 128)),
                   ema_momentum=float(get(cfg, 'mine.ema_momentum', 0.99)), batch_size=int(get(cfg, 'mine.batch_size', 128)),
                   device=resolve_device(cfg))
    seeds = list(get(cfg, 'patching.seeds', [0, 1, 2]))
    n_pairs = int(get(cfg, 'patching.n_pairs', 100))
    var = float(get(cfg, 'representations.pca_variance', 0.95))
    maxk = get(cfg, 'representations.pca_max_components')
    per_step: List[Dict[str, Any]] = []
    for step in [int(s) for s in get(cfg, 'patching.steps', [2, 4, 8])]:
        clean, patched, kept = _states_at_step(model, tokenizer, cfg, benchmark, instances, records, step)
        if clean.shape[0] < 4:
            continue
        answers = [r['answer_pred'] if answer_source == 'model' else r['answer_gold'] for r in kept]
        y = encode_texts(model, tokenizer, answers)
        pca_h = PCAProjector.fit(np.concatenate([clean, patched], axis=0), var, maxk)
        pca_y = PCAProjector.fit(y, var, maxk)
        Hc, Hp, Y = (pca_h.transform(clean), pca_h.transform(patched), pca_y.transform(y))
        if bool(get(cfg, 'representations.standardize', True)):
            sh = Standardizer(np.concatenate([Hc, Hp], axis=0))
            Hc, Hp, Y = (sh(Hc), sh(Hp), Standardizer(Y)(Y))
        vals = []
        for s in seeds:
            idx = mine.bootstrap_indices(Hc.shape[0], min(n_pairs, max(8, Hc.shape[0])), s)
            i_clean = mine.estimate_mi(Hc[idx], Y[idx], seed=s, **mine_kw).score_bits
            i_patched = mine.estimate_mi(Hp[idx], Y[idx], seed=s, **mine_kw).score_bits
            vals.append(i_clean - i_patched)
        arr = np.array(vals, dtype=np.float64)
        per_step.append({'step': step, 'n': int(clean.shape[0]), 'R_mean': float(arr.mean()), 'R_std': float(arr.std(ddof=0)),
                         'per_seed': [float(v) for v in vals]})
        print(f'[patching] {model_key}/{bench_name} step={step} R={arr.mean():.3f}+-{arr.std(ddof=0):.3f}')
    R_vals = np.array([p['R_mean'] for p in per_step], dtype=np.float64)
    R_bar = float(R_vals.mean()) if len(R_vals) else 0.0
    l_proj = theory.projection_loss_bits(dims['v'], dims['d'])
    out = {'model': model_key, 'benchmark': bench_name, 'T_cap': cap, 'n_examples': len(records), 'per_step': per_step,
           'R_bar': R_bar, 'R_min_over_steps': float(R_vals.min()) if len(R_vals) else 0.0, 'R_max_over_steps': float(R_vals.max()) if len(R_vals) else 0.0,
           'l_proj_bits': l_proj, 'dpi_lower_bound_at_R_bar': theory.dpi_lower_bound(cap, dims['v'], dims['d'], R_bar),
           'seeds': seeds, 'mechanism': str(get(cfg, 'patching.mechanism', 'kv_cache'))}
    d = os.path.join(get(cfg, 'paths.results', 'results'), 'patching', model_key, bench_name, f'T{cap}')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'recovery.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out
