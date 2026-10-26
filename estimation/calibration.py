"""Critic calibration on synthetic Gaussian pairs of known mutual information."""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from estimation import mine
from theory import capacity as theory
from experiments.config import get
from experiments.model import resolve_device


def sample_gaussian_pair(n: int, dim: int, mi_bits: float, seed: int):
    rho = theory.rho_for_mi_bits(mi_bits, dim)
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((n, dim)).astype(np.float32)
    eps = rng.standard_normal((n, dim)).astype(np.float32)
    z2 = (rho * z1 + np.sqrt(max(0.0, 1 - rho ** 2)) * eps).astype(np.float32)
    return (z1, z2, rho)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""

    def ranks(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x)
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        for value in np.unique(x):
            tied = x == value
            if tied.sum() > 1:
                r[tied] = r[tied].mean()
        return r
    ra, rb = (ranks(a), ranks(b))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def run_calibration(cfg: Dict[str, Any], out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Critic scores at each target MI, their rank agreement with the truth, and the log2 N ceiling."""
    n = int(get(cfg, 'calibration.n_pairs', 10000))
    dim = int(get(cfg, 'calibration.dim', 247))
    targets = [float(t) for t in get(cfg, 'calibration.target_mi_bits', [0.1, 1.0, 10.0, 100.0])]
    seeds = list(get(cfg, 'calibration.seeds', [0, 1, 2]))
    mine_kw = dict(iterations=int(get(cfg, 'calibration.iterations', 5000)), lr=float(get(cfg, 'mine.lr', 0.0001)),
                   hidden_dims=tuple(get(cfg, 'mine.hidden_dims', [512, 256, 128])), proj_dim=int(get(cfg, 'mine.proj_dim', 128)),
                   ema_momentum=float(get(cfg, 'mine.ema_momentum', 0.99)), batch_size=int(get(cfg, 'mine.batch_size', 128)),
                   device=resolve_device(cfg))
    settings: List[Dict[str, Any]] = []
    for target in targets:
        est = []
        for s in seeds:
            z1, z2, rho = sample_gaussian_pair(n, dim, target, seed=s)
            est.append(mine.estimate_mi(z1, z2, seed=s, **mine_kw).score_bits)
        arr = np.array(est, dtype=np.float64)
        settings.append({'true_mi_bits': target, 'rho': float(rho), 'estimate_mean': float(arr.mean()),
                         'estimate_std': float(arr.std(ddof=0)), 'per_seed': [float(v) for v in est]})
        print(f'[calibration] true={target:g} bits est={arr.mean():.3f}+-{arr.std(ddof=0):.3f} bits')
    rho_rank = spearman([s['true_mi_bits'] for s in settings], [s['estimate_mean'] for s in settings])
    est_sorted = [s['estimate_mean'] for s in sorted(settings, key=lambda s: s['true_mi_bits'])]
    out = {'n_pairs': n, 'dim': dim, 'seeds': seeds, 'settings': settings, 'spearman_rank_correlation': rho_rank,
           'rank_order_preserved': bool(all(est_sorted[i] <= est_sorted[i + 1] for i in range(len(est_sorted) - 1))),
           'dv_ceiling_bits': theory.dv_ceiling_bits(n),
           'orders_of_magnitude': float(np.log10(max(targets) / min(targets))) if min(targets) > 0 else None}
    d = out_dir or os.path.join(get(cfg, 'paths.results', 'results'), 'calibration')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'calibration.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out
