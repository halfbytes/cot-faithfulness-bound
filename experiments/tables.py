"""Aggregation of result files into markdown and JSON tables."""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional
from experiments.config import get

BENCH_LABEL = {'gsm8k': 'GSM8K', 'arc_challenge': 'ARC-Chall.', 'math_algebra': 'MATH-Alg.', 'strategyqa': 'StrategyQA',
               'boardgameqa': 'BoardgameQA', 'gpqa_diamond': 'GPQA-Dia.', 'cryptobench': 'CryptoBench'}


def _read(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _fmt(x, nd=1):
    if x is None:
        return '--'
    return f'{x:.{nd}f}' if isinstance(x, float) else str(x)


def _pm(mean, std, nd=1) -> str:
    return '--' if mean is None else f'{_fmt(mean, nd)}+-{_fmt(std, nd)}'


def gap_path(cfg, model, bench, cap) -> str:
    return os.path.join(get(cfg, 'paths.results', 'results'), 'faithfulness', model, bench, f'T{cap}', 'gap.json')


def ps_path(cfg, model, bench, cap) -> str:
    return os.path.join(get(cfg, 'paths.results', 'results'), 'perturbation', model, bench, f'T{cap}', 'ps.json')


def gen_stats_path(cfg, model, bench, cap) -> str:
    k = int(get(cfg, 'prompting.k_shot', 8))
    return os.path.join(get(cfg, 'paths.results', 'results'), 'generation', model, bench, f'eval_T{cap}_k{k}', 'stats.json')


def _accuracy(cfg, model, bench, cap) -> Optional[float]:
    gen = _read(gen_stats_path(cfg, model, bench, cap))
    return None if not gen else 100 * gen['accuracy']


def main_results(cfg: Dict[str, Any], model_key: str, benchmarks: List[str]) -> Dict[str, Any]:
    """Critic score, capacities, gap, regime and sensitivity per benchmark."""
    from data.registry import generation_cap
    rows = []
    for b in benchmarks:
        cap = generation_cap(cfg, b)
        gap, ps = (_read(gap_path(cfg, model_key, b, cap)), _read(ps_path(cfg, model_key, b, cap)))
        rows.append({'benchmark': b, 'label': BENCH_LABEL.get(b, b), 'T': cap,
                     'I_hat': None if not gap else gap['I_hat_h0_mean'], 'C_TV': None if not gap else gap['C_TV_ceil'],
                     'C_eff': None if not gap else gap['C_eff'], 'delta': None if not gap else gap['delta_mean'],
                     'delta_std': None if not gap else gap['delta_std'], 'regime': None if not gap else gap['regime'],
                     'PS': None if not ps else ps['PS_mean'], 'PS_std': None if not ps else ps['PS_std'],
                     'accuracy': _accuracy(cfg, model_key, b, cap)})
    md = ['| Bench. | T | I-hat | C(T,V) | C_eff | Delta | Regime | PS | Acc % |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['label']} | {r['T']} | {_fmt(r['I_hat'])} | {_fmt(r['C_TV'], 0)} | {_fmt(r['C_eff'], 0)} | "
                  f"{_pm(r['delta'], r['delta_std'])} | {r['regime'] or '--'} | {_fmt(r['PS'], 2)} | {_fmt(r['accuracy'])} |")
    return {'name': 'main_results', 'rows': rows, 'markdown': '\n'.join(md)}


def perturbation_by_model(cfg: Dict[str, Any], model_keys: List[str], benchmarks: List[str]) -> Dict[str, Any]:
    """Sensitivity of each model on each benchmark."""
    from data.registry import generation_cap
    rows = []
    for b in benchmarks:
        cap = generation_cap(cfg, b)
        row = {'benchmark': b, 'label': BENCH_LABEL.get(b, b)}
        for m in model_keys:
            ps = _read(ps_path(cfg, m, b, cap))
            row[m] = None if not ps else ps['PS_mean']
        rows.append(row)
    md = ['| Bench. | ' + ' | '.join(model_keys) + ' |', '|---' * (len(model_keys) + 1) + '|']
    for r in rows:
        md.append('| ' + r['label'] + ' | ' + ' | '.join(_fmt(r[m], 2) for m in model_keys) + ' |')
    return {'name': 'perturbation_by_model', 'rows': rows, 'markdown': '\n'.join(md)}


def complexity_ordering(cfg: Dict[str, Any], model_keys: List[str], benchmarks: List[str]) -> Dict[str, Any]:
    """Critic score, per-token entropy, effective capacity and accuracy of each model per benchmark."""
    from data.registry import generation_cap
    rows = []
    for b in benchmarks:
        cap = generation_cap(cfg, b)
        primary = _read(gap_path(cfg, model_keys[0], b, cap)) if model_keys else None
        row = {'benchmark': b, 'label': BENCH_LABEL.get(b, b), 'I_hat': None if not primary else primary['I_hat_h0_mean'],
               'H_hat': None if not primary else primary['H_hat_bits_per_token'], 'C_eff': None if not primary else primary['C_eff']}
        for m in model_keys:
            row[f'acc_{m}'] = _accuracy(cfg, m, b, cap)
        rows.append(row)
    md = ['| Bench. | I-hat | H-hat | C_eff | ' + ' | '.join(f'acc {m}' for m in model_keys) + ' |',
          '|---|---|---|---|' + '---|' * len(model_keys)]
    for r in rows:
        md.append(f"| {r['label']} | {_fmt(r['I_hat'])} | {_fmt(r['H_hat'], 2)} | {_fmt(r['C_eff'], 0)} | "
                  + ' | '.join(_fmt(r[f'acc_{m}']) for m in model_keys) + ' |')
    return {'name': 'complexity_ordering', 'rows': rows, 'markdown': '\n'.join(md)}


def scaling(cfg: Dict[str, Any], model_key: str, bench: str, caps: List[int]) -> Dict[str, Any]:
    """Gap, sensitivity and accuracy across generation caps."""
    from theory.capacity import predicted_gap
    ref_cap = get(cfg, 'scaling.pred_reference_cap')
    ref_gap = _read(gap_path(cfg, model_key, bench, int(ref_cap))) if ref_cap else None
    ref_I = ref_gap['I_hat_h0_mean'] if ref_gap else None
    if ref_cap and ref_I is None:
        print(f'[tables] scaling: pred_reference_cap={ref_cap} set but gap.json absent; falling back to per-cap Pred.')
    rows = []
    for cap in caps:
        gap, ps = (_read(gap_path(cfg, model_key, bench, cap)), _read(ps_path(cfg, model_key, bench, cap)))
        if ref_I is not None and gap:
            pred = predicted_gap(ref_I, cap, gap['dims']['v'])
        else:
            pred = None if not gap else gap['predicted_gap_ordering_device']
        rows.append({'T': cap, 'C_TV': None if not gap else gap['C_TV_ceil'],
                     'pred': pred, 'pred_reference_cap': int(ref_cap) if ref_I is not None else None,
                     'delta': None if not gap else gap['delta_mean'], 'delta_std': None if not gap else gap['delta_std'],
                     'PS': None if not ps else ps['PS_mean'], 'accuracy': _accuracy(cfg, model_key, bench, cap)})
    md = ['| Max T | C(T,V) | Pred. | Delta | PS | Acc % |', '|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['T']} | {_fmt(r['C_TV'], 0)} | {_fmt(r['pred'], 0)} | {_pm(r['delta'], r['delta_std'])} | "
                  f"{_fmt(r['PS'], 2)} | {_fmt(r['accuracy'])} |")
    return {'name': 'scaling', 'rows': rows, 'markdown': '\n'.join(md)}


def recovery_sensitivity(T: int, v: int, d: int, measured_delta: float, R_grid: List[float]) -> Dict[str, Any]:
    """Predicted bound against the measured gap over a grid of recovery values."""
    from theory.capacity import dpi_sensitivity_table
    rows = dpi_sensitivity_table(T, v, d, R_grid, measured_delta)
    md = ['| R-bar (bits) | Predicted Delta | Measured Delta | Satisfied |', '|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['R_bar']} | {r['pred_delta']} | {_fmt(r['measured_delta'])} | {'yes' if r['satisfied'] else 'no'} |")
    return {'name': 'recovery_sensitivity', 'rows': rows, 'markdown': '\n'.join(md)}


def cross_model(cfg: Dict[str, Any], model_keys: List[str], benchmarks: List[str]) -> Dict[str, Any]:
    """Gap and capacity of each model on each benchmark."""
    from data.registry import generation_cap
    rows = []
    for b in benchmarks:
        cap = generation_cap(cfg, b)
        for m in model_keys:
            gap = _read(gap_path(cfg, m, b, cap))
            rows.append({'benchmark': b, 'model': m, 'C_TV': None if not gap else gap['C_TV_ceil'],
                         'delta': None if not gap else gap['delta_mean'], 'delta_std': None if not gap else gap['delta_std'],
                         'accuracy': _accuracy(cfg, m, b, cap)})
    md = ['| Bench. | Model | C(T,V) | Delta | Acc % |', '|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {BENCH_LABEL.get(r['benchmark'], r['benchmark'])} | {r['model']} | {_fmt(r['C_TV'], 0)} | "
                  f"{_pm(r['delta'], r['delta_std'])} | {_fmt(r['accuracy'])} |")
    return {'name': 'cross_model', 'rows': rows, 'markdown': '\n'.join(md)}


def frontier(cfg: Dict[str, Any], model_key: str) -> Optional[Dict[str, Any]]:
    """Sensitivity and accuracy at each penalty weight."""
    data = _read(os.path.join(get(cfg, 'paths.results', 'results'), 'ib', model_key, 'frontier.json'))
    if not data:
        return None
    md = ['| Benchmark | Setting | PS | Acc % |', '|---|---|---|---|']
    rows = []
    for bench, points in data.items():
        for p in points:
            rows.append({**p, 'benchmark': bench})
            md.append(f"| {BENCH_LABEL.get(bench, bench)} | {p['label']} | {_fmt(p['PS'], 2)} | {_fmt(p['accuracy'])} |")
    return {'name': 'frontier', 'rows': rows, 'markdown': '\n'.join(md)}


def estimator_agreement(cfg: Dict[str, Any], model_key: str) -> Optional[Dict[str, Any]]:
    """Both estimators' scores per benchmark and their rank correlation."""
    data = _read(os.path.join(get(cfg, 'paths.results', 'results'), 'estimators', model_key, 'comparison.json'))
    if not data:
        return None
    md = ['| Benchmark | MINE I(y;h0) | MINE Delta | InfoNCE I(y;h0) | InfoNCE Delta |', '|---|---|---|---|---|']
    for r in data['rows']:
        md.append(f"| {BENCH_LABEL.get(r['benchmark'], r['benchmark'])} | {_fmt(r['mine_h0_mean'], 2)} | {_fmt(r['mine_delta_mean'], 2)} | "
                  f"{_fmt(r['infonce_h0_mean'], 2)} | {_fmt(r['infonce_delta_mean'], 2)} |")
    md.append('')
    md.append(f"Spearman rank correlation of the gaps: {data['spearman_rank_correlation']:.2f}; gap ordering preserved: "
              f"{data['rank_order_preserved']}; I(y;h0) ordering preserved: {data['rank_order_preserved_h0']}; "
              f"InfoNCE ceiling: {data['infonce_ceiling_bits']:.1f} bits.")
    return {'name': 'estimator_agreement', 'rows': data['rows'], 'markdown': '\n'.join(md)}


def write_tables(cfg: Dict[str, Any], tables: List[Optional[Dict[str, Any]]], out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or os.path.join(get(cfg, 'paths.results', 'results'), 'tables')
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    for t in tables:
        if not t:
            continue
        with open(os.path.join(out_dir, f"{t['name']}.json"), 'w', encoding='utf-8') as f:
            json.dump(t['rows'], f, indent=2)
        parts.append(f"## {t['name']}\n\n{t['markdown']}\n")
    path = os.path.join(out_dir, 'tables.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Results\n\n' + '\n'.join(parts))
    return path
