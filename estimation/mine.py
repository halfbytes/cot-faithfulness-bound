"""Donsker-Varadhan (MINE) and contrastive (InfoNCE) critic scores."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

LN2 = math.log(2.0)


class Critic(nn.Module):
    """Per-input linear projections followed by an MLP that scores a pair."""

    def __init__(self, dim_u: int, dim_w: int, hidden_dims: Sequence[int] = (512, 256, 128), proj_dim: int = 128):
        super().__init__()
        self.proj_u = nn.Linear(dim_u, proj_dim)
        self.proj_w = nn.Linear(dim_w, proj_dim)
        layers: List[nn.Module] = []
        prev = 2 * proj_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.proj_u(u), self.proj_w(w)], dim=-1)
        return self.mlp(z).squeeze(-1)


@dataclass
class MineResult:
    score_bits: float
    score_nats: float
    seed: int
    iterations: int
    curve: List[Dict[str, float]] = field(default_factory=list)
    n_pairs: int = 0


def bootstrap_indices(n: int, n_pairs: int, seed: int) -> np.ndarray:
    """n_pairs indices drawn with replacement from range(n)."""
    if n <= 0:
        raise ValueError('cannot resample from an empty evaluation set')
    return np.random.default_rng(seed).integers(0, n, size=n_pairs)


def dv_score(t_joint: torch.Tensor, t_marginal: torch.Tensor) -> torch.Tensor:
    """Donsker-Varadhan value in nats: E_P[T] - log E_Q[e^T]."""
    return t_joint.mean() - (torch.logsumexp(t_marginal, dim=0) - math.log(t_marginal.shape[0]))


def estimate_mi(U: np.ndarray, W: np.ndarray, seed: int = 0, iterations: int = 5000, lr: float = 0.0001,
                hidden_dims: Sequence[int] = (512, 256, 128), proj_dim: int = 128, ema_momentum: float = 0.99,
                batch_size: int = 128, eval_every: int = 0, device: Optional[torch.device] = None) -> MineResult:
    """Train a critic on paired samples (U[i], W[i]) with the moving-average gradient correction; return the DV score in bits."""
    if U.shape[0] != W.shape[0]:
        raise ValueError(f'paired samples must have equal length, got {U.shape[0]} and {W.shape[0]}')
    n = U.shape[0]
    device = device or torch.device('cpu')
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    torch.manual_seed(seed)
    u = torch.tensor(np.asarray(U, dtype=np.float32), device=device)
    w = torch.tensor(np.asarray(W, dtype=np.float32), device=device)
    critic = Critic(u.shape[1], w.shape[1], hidden_dims=hidden_dims, proj_dim=proj_dim).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    ema: Optional[torch.Tensor] = None
    bs = min(batch_size, n)
    curve: List[Dict[str, float]] = []
    for it in range(1, iterations + 1):
        idx = torch.randint(0, n, (bs,), generator=g)
        shuffle = torch.randperm(bs, generator=g)
        ub, wb = (u[idx], w[idx])
        t_joint = critic(ub, wb)
        t_marg = critic(ub, wb[shuffle])
        exp_marg = torch.exp(t_marg).mean()
        ema = exp_marg.detach() if ema is None else ema_momentum * ema + (1 - ema_momentum) * exp_marg.detach()
        loss = -(t_joint.mean() - exp_marg / ema.clamp_min(1e-12))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if eval_every and (it % eval_every == 0 or it == iterations):
            curve.append({'iteration': it, 'score_bits': _full_sample_score(critic, u, w, g) / LN2})
    nats = _full_sample_score(critic, u, w, g)
    return MineResult(score_bits=nats / LN2, score_nats=nats, seed=seed, iterations=iterations, curve=curve, n_pairs=n)


@torch.no_grad()
def _full_sample_score(critic: Critic, u: torch.Tensor, w: torch.Tensor, g: torch.Generator) -> float:
    """DV value in nats on all pairs, with a fixed-point-free shuffle of w as the marginal."""
    n = u.shape[0]
    perm = torch.randperm(n, generator=g)
    fixed = perm == torch.arange(n)
    if fixed.any() and n > 1:
        perm[fixed] = (perm[fixed] + 1) % n
    return float(dv_score(critic(u, w), critic(u, w[perm])).item())


def infonce_score(U: np.ndarray, W: np.ndarray, seed: int = 0, iterations: int = 5000, lr: float = 0.0001,
                  hidden_dims: Sequence[int] = (512, 256, 128), proj_dim: int = 128, batch_size: int = 128,
                  device: Optional[torch.device] = None) -> MineResult:
    if U.shape[0] != W.shape[0]:
        raise ValueError(f'paired samples must have equal length, got {U.shape[0]} and {W.shape[0]}')
    n = U.shape[0]
    device = device or torch.device('cpu')
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    torch.manual_seed(seed)
    u = torch.tensor(np.asarray(U, dtype=np.float32), device=device)
    w = torch.tensor(np.asarray(W, dtype=np.float32), device=device)
    critic = Critic(u.shape[1], w.shape[1], hidden_dims=hidden_dims, proj_dim=proj_dim).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    bs = min(batch_size, n)
    for _ in range(iterations):
        idx = torch.randint(0, n, (bs,), generator=g)
        ub, wb = (u[idx], w[idx])
        scores = critic(ub.repeat_interleave(bs, 0), wb.repeat(bs, 1)).view(bs, bs)
        loss = torch.nn.functional.cross_entropy(scores, torch.arange(bs, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        idx = torch.randint(0, n, (bs,), generator=g)
        ub, wb = (u[idx], w[idx])
        scores = critic(ub.repeat_interleave(bs, 0), wb.repeat(bs, 1)).view(bs, bs)
        ce = torch.nn.functional.cross_entropy(scores, torch.arange(bs, device=device))
        nats = float(math.log(bs) - ce.item())
    return MineResult(score_bits=nats / LN2, score_nats=nats, seed=seed, iterations=iterations, n_pairs=n)


def infonce_ceiling_bits(batch_size: int) -> float:
    """log2 of the batch size."""
    return math.log2(batch_size)


def convergence_window_std(curve: Sequence[Dict[str, float]], start: int, end: int) -> Dict[str, float]:
    """Spread of the critic score over an iteration window, absolute and relative to the final value."""
    vals = [c['score_bits'] for c in curve if start <= c['iteration'] <= end]
    if not vals:
        return {'n': 0, 'std': 0.0, 'fraction_of_final': 0.0, 'start': start, 'end': end}
    final = curve[-1]['score_bits']
    std = float(np.std(vals, ddof=0))
    return {'n': len(vals), 'std': std, 'mean': float(np.mean(vals)), 'fraction_of_final': std / abs(final) if final else 0.0,
            'start': start, 'end': end}


def convergence_iteration(curve: Sequence[Dict[str, float]], tol_frac: float = 0.011, window: int = 2) -> Optional[int]:
    """First recorded iteration at which `window` consecutive recorded scores lie within tol_frac of the final score."""
    if not curve:
        return None
    final = curve[-1]['score_bits']
    tol = abs(final) * tol_frac
    for i in range(len(curve)):
        if all(abs(c['score_bits'] - final) <= tol for c in curve[i:i + max(1, window)]):
            return int(curve[i]['iteration'])
    return int(curve[-1]['iteration'])
