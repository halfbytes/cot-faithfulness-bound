"""Closed-form channel-capacity quantities (all logarithms base 2)."""
from __future__ import annotations
import math
from typing import Dict, Iterable, List, Sequence


def log2_vocab(v: int) -> float:
    if v <= 1:
        raise ValueError('vocabulary size must exceed 1')
    return math.log2(v)


def capacity_bits(T: int, v: int) -> float:
    """Maximum trace capacity T * log2|V|."""
    if T < 0:
        raise ValueError('T must be non-negative')
    return T * log2_vocab(v)


def capacity_bits_ceil(T: int, v: int) -> int:
    """Whole-bit capacity T * ceil(log2|V|)."""
    return int(T * math.ceil(log2_vocab(v)))


def effective_capacity(T: int, h_hat_bits_per_token: float) -> float:
    """Effective capacity T * Hhat for a per-token entropy estimate Hhat."""
    if h_hat_bits_per_token < 0:
        raise ValueError('per-token entropy must be non-negative')
    return T * h_hat_bits_per_token


def projection_loss_bits(v: int, d: int) -> float:
    """Per-step projection loss log2(v/d)."""
    if d <= 0 or v <= 0:
        raise ValueError('v and d must be positive')
    return math.log2(v / d)


def dpi_lower_bound(T: int, v: int, d: int, R: Sequence[float] | float) -> float:
    """max(0, T * l_proj - sum of the per-step recovery R)."""
    if isinstance(R, (int, float)):
        total_R = T * float(R)
    else:
        R = list(R)
        if len(R) != T:
            raise ValueError(f'per-step R must have length T={T}, got {len(R)}')
        total_R = float(sum(R))
    return max(0.0, T * projection_loss_bits(v, d) - total_R)


def dpi_prediction(T: int, v: int, d: int, R_bar: float) -> int:
    return int(round(dpi_lower_bound(T, v, d, R_bar)))


def dpi_sensitivity_table(T: int, v: int, d: int, R_grid: Iterable[float], measured_delta: float) -> List[Dict]:
    rows = []
    for R in R_grid:
        pred = dpi_prediction(T, v, d, R)
        rows.append({'R_bar': R, 'pred_delta': pred, 'measured_delta': measured_delta, 'satisfied': measured_delta >= pred})
    return rows


def predicted_gap(I_hat: float, T: int, v: int) -> float:
    """max(0, Ihat - T * ceil(log2|V|))."""
    return max(0.0, I_hat - capacity_bits_ceil(T, v))


def regime(I_hat: float, C_eff: float) -> str:
    """'above' if the critic score exceeds the effective capacity, else 'within'."""
    return 'above' if I_hat > C_eff else 'within'


def bound_binds(answer_len_tokens: int, T: int) -> bool:
    """Whether the answer is longer than the trace cap."""
    return answer_len_tokens > T


def ps_upper_bound(n_answers: int) -> float:
    """1 - 1/|A| for an answer space of size |A|."""
    if n_answers < 1:
        raise ValueError('answer space size must be positive')
    return 1.0 - 1.0 / n_answers


def dv_ceiling_bits(n_samples: int) -> float:
    """log2 N."""
    return math.log2(n_samples)


def rho_for_mi_bits(mi_bits: float, dim: int) -> float:
    """Correlation of z2 = rho*z1 + sqrt(1-rho^2)*eps giving the requested MI between dim-dimensional Gaussians."""
    if mi_bits < 0:
        raise ValueError('MI must be non-negative')
    return math.sqrt(1.0 - 2.0 ** (-2.0 * mi_bits / dim))


def binomial_tail(k: int, n: int, p0: float) -> float:
    """Exact one-sided binomial tail P[X >= k] under success probability p0."""
    from math import comb
    return float(sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1)))
