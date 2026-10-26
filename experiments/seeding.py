"""Seeding of every stochastic source."""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    g = torch.Generator()
    g.manual_seed(seed)
    return g
