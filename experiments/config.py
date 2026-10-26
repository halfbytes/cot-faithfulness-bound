"""YAML configuration loading with inheritance and dotted overrides."""
from __future__ import annotations
import copy
import os
from typing import Any, Dict, Iterable, Optional
import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config {path} must be a mapping at top level')
    return data


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a config file, resolving the _base_ list and applying key.sub=value overrides."""
    path = os.path.abspath(path)
    data = _load_yaml(path)
    bases = data.pop('_base_', None)
    merged: Dict[str, Any] = {}
    if bases:
        if isinstance(bases, str):
            bases = [bases]
        for b in bases:
            bpath = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
            merged = deep_merge(merged, load_config(bpath))
    merged = deep_merge(merged, data)
    merged.setdefault('_config_path', path)
    for ov in overrides or []:
        merged = apply_override(merged, ov)
    return merged


def apply_override(cfg: Dict[str, Any], override: str) -> Dict[str, Any]:
    if '=' not in override:
        raise ValueError(f'Override must look like key.sub=value, got {override!r}')
    key, raw = override.split('=', 1)
    value = yaml.safe_load(raw)
    parts = key.strip().split('.')
    out = copy.deepcopy(cfg)
    node = out
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value
    return out


def get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read cfg by dotted path with a default."""
    node: Any = cfg
    for p in dotted.split('.'):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
