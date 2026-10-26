"""Critic-score gap measurement, per-category gaps, sample-size sensitivity and the estimator comparison."""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from estimation import mine
from theory import capacity as theory
from experiments.config import get
from experiments.generation import generation_dir, load_generation, load_states, run_generation
from experiments.model import load_model_and_tokenizer, model_dims, resolve_device
from estimation.representations import PCAProjector, Standardizer, encode_texts, split_holdout
from experiments.seeding import seed_everything


def faithfulness_dir(cfg: Dict[str, Any], model_key: str, bench: str, cap: int) -> str:
    return os.path.join(get(cfg, 'paths.results', 'results'), 'faithfulness', model_key, bench, f'T{cap}')


def _ensure_generation(cfg, model_key, bench, split, cap, model_and_tok, **kw) -> str:
    d = generation_dir(cfg, model_key, bench, split, cap)
    if os.path.exists(os.path.join(d, 'traces.jsonl')) and not get(cfg, 'force_regenerate', False):
        return d
    run_generation(cfg, model_key, bench, split, cap=cap, model_and_tok=model_and_tok, **kw)
    return d


def _mine_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(iterations=int(get(cfg, 'mine.iterations', 5000)), lr=float(get(cfg, 'mine.lr', 0.0001)),
                hidden_dims=tuple(get(cfg, 'mine.hidden_dims', [512, 256, 128])), proj_dim=int(get(cfg, 'mine.proj_dim', 128)),
                ema_momentum=float(get(cfg, 'mine.ema_momentum', 0.99)), batch_size=int(get(cfg, 'mine.batch_size', 128)),
                eval_every=int(get(cfg, 'mine.eval_every', 0)), device=resolve_device(cfg))


def build_inputs(cfg: Dict[str, Any], model, tokenizer, records: List[Dict[str, Any]], h0: np.ndarray,
                 holdout_h0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Encode the prompt-final states, answers and traces, project each with PCA and standardise."""
    var = float(get(cfg, 'representations.pca_variance', 0.95))
    maxk = get(cfg, 'representations.pca_max_components')
    answer_source = str(get(cfg, 'representations.answer_source', 'model'))
    answers = [r['answer_pred'] if answer_source == 'model' else r['answer_gold'] for r in records]
    traces = [r['trace_text'] for r in records]
    y_raw = encode_texts(model, tokenizer, answers)
    c_raw = encode_texts(model, tokenizer, traces)
    fit_h0 = holdout_h0 if holdout_h0 is not None and holdout_h0.shape[0] >= 2 else h0
    pca_h = PCAProjector.fit(fit_h0, var, maxk)
    pca_y = PCAProjector.fit(y_raw, var, maxk)
    pca_c = PCAProjector.fit(c_raw, var, maxk)
    out = {'h0': pca_h.transform(h0), 'y': pca_y.transform(y_raw), 'c': pca_c.transform(c_raw)}
    if bool(get(cfg, 'representations.standardize', True)):
        out = {k: Standardizer(v)(v) for k, v in out.items()}
    out['_dims'] = np.array([pca_h.n_components, pca_y.n_components, pca_c.n_components])
    return out


def _window(cfg: Dict[str, Any], curve) -> Dict[str, float]:
    lo, hi = get(cfg, 'mine.convergence_window', [3000, 5000])
    return mine.convergence_window_std(curve, int(lo), int(hi))


def _gap_curve(curve_h0, curve_c) -> List[Dict[str, float]]:
    """Per-iteration difference of two critic-score curves recorded at the same iterations."""
    return [{'iteration': a['iteration'], 'score_bits': a['score_bits'] - b['score_bits']} for a, b in zip(curve_h0, curve_c)]


def measure_gap(cfg: Dict[str, Any], model_key: str, bench_name: str, cap: Optional[int] = None, model_and_tok=None,
                tag: str = '') -> Dict[str, Any]:
    """Critic scores I(y;h0) and I(y;c), their gap, capacities and regime label for one (model, benchmark)."""
    from data.registry import generation_cap
    seed_everything(int(get(cfg, 'seed', 0)))
    cap = int(cap or generation_cap(cfg, bench_name))
    model, tokenizer = model_and_tok or load_model_and_tokenizer(cfg, cfg['models'][model_key])
    dims = model_dims(model, tokenizer)
    eval_dir = _ensure_generation(cfg, model_key, bench_name, 'eval', cap, (model, tokenizer))
    records, h0 = load_generation(eval_dir)
    with open(os.path.join(eval_dir, 'stats.json'), 'r', encoding='utf-8') as f:
        gen_stats = json.load(f)
    try:
        val_dir = _ensure_generation(cfg, model_key, bench_name, 'validation', cap, (model, tokenizer))
        _, holdout_h0 = load_generation(val_dir)
        holdout_states = load_states(val_dir)
        if holdout_states is not None:
            holdout_h0 = holdout_states
    except FileNotFoundError:
        hidx, rest = split_holdout(h0.shape[0], max(2, h0.shape[0] // 5), int(get(cfg, 'seed', 0)))
        holdout_h0 = h0[hidx]
        records, h0 = [records[i] for i in rest], h0[rest]
    inputs = build_inputs(cfg, model, tokenizer, records, h0, holdout_h0)
    mine_kw = _mine_kwargs(cfg)
    seeds = list(get(cfg, 'mine.seeds', [0, 1, 2, 3, 4]))
    n_pairs = int(get(cfg, 'mine.n_pairs', 1000))
    per_seed = []
    for s in seeds:
        idx = mine.bootstrap_indices(len(records), n_pairs, s)
        r_h = mine.estimate_mi(inputs['h0'][idx], inputs['y'][idx], seed=s, **mine_kw)
        r_c = mine.estimate_mi(inputs['c'][idx], inputs['y'][idx], seed=s, **mine_kw)
        per_seed.append({'seed': int(s), 'I_h0': r_h.score_bits, 'I_c': r_c.score_bits, 'delta': r_h.score_bits - r_c.score_bits,
                         'curve_h0': r_h.curve, 'curve_c': r_c.curve,
                         'convergence_iteration_h0': mine.convergence_iteration(r_h.curve, float(get(cfg, 'mine.convergence_tol_frac', 0.011))),
                         'convergence_iteration_c': mine.convergence_iteration(r_c.curve, float(get(cfg, 'mine.convergence_tol_frac', 0.011))),
                         'convergence_window_h0': _window(cfg, r_h.curve), 'convergence_window_c': _window(cfg, r_c.curve),
                         'convergence_window_delta': _window(cfg, _gap_curve(r_h.curve, r_c.curve))})
        print(f'[mine] {model_key}/{bench_name} seed={s} I(y;h0)={r_h.score_bits:.2f} I(y;c)={r_c.score_bits:.2f} delta={r_h.score_bits - r_c.score_bits:.2f}')
    I_h = np.array([p['I_h0'] for p in per_seed])
    I_c = np.array([p['I_c'] for p in per_seed])
    D = np.array([p['delta'] for p in per_seed])
    H_hat = gen_stats['H_hat_bits_per_token']
    C_eff = theory.effective_capacity(cap, H_hat)
    out = {'model': model_key, 'benchmark': bench_name, 'T_cap': cap, 'N': len(records), 'dims': dims,
           'I_hat_h0_mean': float(I_h.mean()), 'I_hat_h0_std': float(I_h.std(ddof=0)),
           'I_hat_c_mean': float(I_c.mean()), 'I_hat_c_std': float(I_c.std(ddof=0)),
           'delta_mean': float(D.mean()), 'delta_std': float(D.std(ddof=0)),
           'H_hat_bits_per_token': H_hat, 'C_eff': C_eff, 'C_TV_ceil': theory.capacity_bits_ceil(cap, dims['v']),
           'C_TV': theory.capacity_bits(cap, dims['v']), 'regime': theory.regime(float(I_h.mean()), C_eff),
           'predicted_gap_ordering_device': theory.predicted_gap(float(I_h.mean()), cap, dims['v']),
           'dv_ceiling_bits': theory.dv_ceiling_bits(len(records)), 'accuracy': gen_stats['accuracy'],
           'mean_trace_tokens': gen_stats['mean_trace_tokens'], 'mean_answer_tokens': gen_stats['mean_answer_tokens'],
           'effective_dim_h0': int(inputs['_dims'][0]), 'effective_dim_y': int(inputs['_dims'][1]),
           'effective_dim_c': int(inputs['_dims'][2]),
           'bound_binds': theory.bound_binds(int(round(gen_stats['mean_answer_tokens'])), cap),
           'per_seed': per_seed, 'seeds': seeds, 'n_pairs': n_pairs}
    d = faithfulness_dir(cfg, model_key, bench_name, cap) + tag
    os.makedirs(d, exist_ok=True)
    out['per_category'] = per_category_gaps(cfg, model_key, bench_name, records, inputs, mine_kw, seeds, n_pairs)
    with open(os.path.join(d, 'gap.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out


def per_category_gaps(cfg: Dict[str, Any], model_key: str, bench_name: str, records: List[Dict[str, Any]],
                      inputs: Dict[str, np.ndarray], mine_kw: Dict[str, Any], seeds: Sequence[int], n_pairs: int) -> Dict[str, Any]:
    from data.registry import get_benchmark
    benchmark = get_benchmark(cfg, bench_name)
    try:
        cat_of = {i.id: i.meta.get('category') for i in benchmark.load('eval', limit=get(cfg, 'generation.limit'))}
    except FileNotFoundError:
        return {}
    rows: Dict[str, List[int]] = {}
    for j, r in enumerate(records):
        c = cat_of.get(r['id'])
        if c:
            rows.setdefault(c, []).append(j)
    min_n = int(get(cfg, 'mine.min_category_pairs', 16))
    out: Dict[str, Any] = {}
    for cat, idx in sorted(rows.items()):
        if len(idx) < min_n:
            out[cat] = {'n': len(idx), 'skipped': f'fewer than {min_n} instances'}
            continue
        arr = np.array(idx)
        cat_pairs = min(n_pairs, max(len(arr), min_n))
        per_seed = []
        for s in seeds:
            take = arr[mine.bootstrap_indices(len(arr), cat_pairs, s)]
            i_h = mine.estimate_mi(inputs['h0'][take], inputs['y'][take], seed=s, **mine_kw).score_bits
            i_c = mine.estimate_mi(inputs['c'][take], inputs['y'][take], seed=s, **mine_kw).score_bits
            per_seed.append({'seed': int(s), 'I_h0': i_h, 'I_c': i_c, 'delta': i_h - i_c})
        D = np.array([p['delta'] for p in per_seed])
        H = np.array([p['I_h0'] for p in per_seed])
        out[cat] = {'n': len(idx), 'n_pairs': int(cat_pairs), 'I_hat_h0_mean': float(H.mean()), 'delta_mean': float(D.mean()),
                    'delta_std': float(D.std(ddof=0)), 'per_seed': per_seed,
                    'accuracy': float(np.mean([records[j]['correct'] for j in idx]))}
        print(f'[mine] {model_key}/{bench_name}/{cat} n={len(idx)} I={H.mean():.2f} delta={D.mean():.2f}')
    return out


def sample_size_sensitivity(cfg: Dict[str, Any], model_key: str, bench_name: str, sizes: Sequence[int],
                            cap: Optional[int] = None, model_and_tok=None) -> List[Dict[str, Any]]:
    """Critic scores as a function of the number of sample pairs."""
    from data.registry import generation_cap
    cap = int(cap or generation_cap(cfg, bench_name))
    out = []
    for n in sizes:
        cfg_n = json.loads(json.dumps({k: v for k, v in cfg.items() if not k.startswith('_')}))
        cfg_n['mine']['n_pairs'] = int(n)
        r = measure_gap(cfg_n, model_key, bench_name, cap=cap, model_and_tok=model_and_tok, tag=f'_n{n}')
        out.append({'n_pairs': int(n), 'I_hat_h0_mean': r['I_hat_h0_mean'], 'I_hat_h0_std': r['I_hat_h0_std'],
                    'delta_mean': r['delta_mean'], 'delta_std': r['delta_std']})
    d = faithfulness_dir(cfg, model_key, bench_name, cap)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'sample_size.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out


def estimator_comparison(cfg: Dict[str, Any], model_key: str, bench_names: Sequence[str], model_and_tok=None) -> Dict[str, Any]:
    """Donsker-Varadhan and contrastive critic scores and gaps per benchmark, and their rank agreement."""
    from estimation.calibration import spearman
    from data.registry import generation_cap
    model, tokenizer = model_and_tok or load_model_and_tokenizer(cfg, cfg['models'][model_key])
    seeds = list(get(cfg, 'mine.seeds', [0, 1, 2, 3, 4]))
    n_pairs = int(get(cfg, 'mine.n_pairs', 1000))
    batch_size = int(get(cfg, 'mine.batch_size', 128))
    common = dict(lr=float(get(cfg, 'mine.lr', 0.0001)), hidden_dims=tuple(get(cfg, 'mine.hidden_dims', [512, 256, 128])),
                  proj_dim=int(get(cfg, 'mine.proj_dim', 128)), batch_size=batch_size, device=resolve_device(cfg))
    dv_kw = dict(iterations=int(get(cfg, 'mine.iterations', 5000)), ema_momentum=float(get(cfg, 'mine.ema_momentum', 0.99)), **common)
    nce_kw = dict(iterations=int(get(cfg, 'infonce.iterations', get(cfg, 'mine.iterations', 5000))), **common)
    rows = []
    for bench in bench_names:
        cap = generation_cap(cfg, bench)
        eval_dir = _ensure_generation(cfg, model_key, bench, 'eval', cap, (model, tokenizer))
        records, h0 = load_generation(eval_dir)
        try:
            val_dir = _ensure_generation(cfg, model_key, bench, 'validation', cap, (model, tokenizer))
            _, holdout_h0 = load_generation(val_dir)
            holdout_states = load_states(val_dir)
            if holdout_states is not None:
                holdout_h0 = holdout_states
        except FileNotFoundError:
            hidx, rest = split_holdout(h0.shape[0], max(2, h0.shape[0] // 5), int(get(cfg, 'seed', 0)))
            holdout_h0 = h0[hidx]
            records, h0 = [records[i] for i in rest], h0[rest]
        inputs = build_inputs(cfg, model, tokenizer, records, h0, holdout_h0)
        scores = {k: [] for k in ('mine_h0', 'mine_c', 'infonce_h0', 'infonce_c')}
        for s_ in seeds:
            idx = mine.bootstrap_indices(len(records), n_pairs, s_)
            scores['mine_h0'].append(mine.estimate_mi(inputs['h0'][idx], inputs['y'][idx], seed=s_, **dv_kw).score_bits)
            scores['mine_c'].append(mine.estimate_mi(inputs['c'][idx], inputs['y'][idx], seed=s_, **dv_kw).score_bits)
            scores['infonce_h0'].append(mine.infonce_score(inputs['h0'][idx], inputs['y'][idx], seed=s_, **nce_kw).score_bits)
            scores['infonce_c'].append(mine.infonce_score(inputs['c'][idx], inputs['y'][idx], seed=s_, **nce_kw).score_bits)
        row = {'benchmark': bench}
        for k, v in scores.items():
            row[f'{k}_mean'], row[f'{k}_std'] = (float(np.mean(v)), float(np.std(v)))
        for est in ('mine', 'infonce'):
            deltas = np.array(scores[f'{est}_h0']) - np.array(scores[f'{est}_c'])
            row[f'{est}_delta_mean'], row[f'{est}_delta_std'] = (float(deltas.mean()), float(deltas.std()))
        rows.append(row)
        print(f"[estimators] {bench}: MINE I(y;h0) {row['mine_h0_mean']:.2f} delta {row['mine_delta_mean']:.2f} | "
              f"InfoNCE I(y;h0) {row['infonce_h0_mean']:.2f} delta {row['infonce_delta_mean']:.2f} "
              f'(ceiling {mine.infonce_ceiling_bits(batch_size):.1f} bits)')
    rankings = {key: [r['benchmark'] for r in sorted(rows, key=lambda r: r[f'{key}_mean'])]
                for key in ('mine_delta', 'infonce_delta', 'mine_h0', 'infonce_h0')}
    out = {'model': model_key, 'rows': rows, 'seeds': seeds, 'infonce_ceiling_bits': mine.infonce_ceiling_bits(batch_size),
           'spearman_rank_correlation': spearman([r['mine_delta_mean'] for r in rows], [r['infonce_delta_mean'] for r in rows]),
           'spearman_rank_correlation_h0': spearman([r['mine_h0_mean'] for r in rows], [r['infonce_h0_mean'] for r in rows]),
           'ranking_mine': rankings['mine_delta'], 'ranking_infonce': rankings['infonce_delta'],
           'ranking_mine_h0': rankings['mine_h0'], 'ranking_infonce_h0': rankings['infonce_h0'],
           'rank_order_preserved': rankings['mine_delta'] == rankings['infonce_delta'],
           'rank_order_preserved_h0': rankings['mine_h0'] == rankings['infonce_h0']}
    d = os.path.join(get(cfg, 'paths.results', 'results'), 'estimators', model_key)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out
