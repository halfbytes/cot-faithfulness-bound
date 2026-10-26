"""Vector representations fed to the critic: mean input embeddings, PCA projection, standardisation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import torch


@torch.no_grad()
def mean_input_embeddings(model, token_id_lists: Sequence[Sequence[int]]) -> np.ndarray:
    """(N, d) mean of the input-embedding rows of each token list; zeros for an empty list."""
    emb = model.get_input_embeddings().weight.detach()
    d = emb.shape[1]
    out = np.zeros((len(token_id_lists), d), dtype=np.float32)
    for i, ids in enumerate(token_id_lists):
        ids = [int(t) for t in ids if 0 <= int(t) < emb.shape[0]]
        if ids:
            out[i] = emb[torch.tensor(ids, device=emb.device)].float().mean(dim=0).cpu().numpy()
    return out


def encode_texts(model, tokenizer, texts: Sequence[str]) -> np.ndarray:
    ids = [tokenizer.encode(t, add_special_tokens=False) if t else [] for t in texts]
    return mean_input_embeddings(model, ids)


@dataclass
class PCAProjector:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    total_variance: float

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @staticmethod
    def fit(X: np.ndarray, variance: float = 0.95, max_components: Optional[int] = None) -> 'PCAProjector':
        """Smallest number of components whose cumulative explained variance reaches the target."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError('PCA needs a 2-D array with at least 2 rows')
        mean = X.mean(axis=0)
        Xc = X - mean
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        var = s ** 2 / max(1, X.shape[0] - 1)
        total = float(var.sum())
        if total <= 0:
            k = 1
            ratio = np.ones(1)
        else:
            ratio = var / total
            cum = np.cumsum(ratio)
            k = int(np.searchsorted(cum, variance - 1e-12) + 1)
            k = max(1, min(k, len(s)))
        if max_components is not None:
            k = min(k, int(max_components))
        return PCAProjector(mean=mean.astype(np.float32), components=vt[:k].astype(np.float32),
                            explained_variance_ratio=ratio[:k].astype(np.float32), total_variance=total)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float32) - self.mean) @ self.components.T).astype(np.float32)


class Standardizer:
    """Per-feature z-scoring."""

    def __init__(self, X: np.ndarray, eps: float = 1e-06):
        X = np.asarray(X, dtype=np.float32)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + eps

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


def split_holdout(n: int, n_holdout: int, seed: int):
    """(holdout indices, remaining indices) of a seeded permutation of range(n)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return (np.sort(idx[:n_holdout]), np.sort(idx[n_holdout:]))
