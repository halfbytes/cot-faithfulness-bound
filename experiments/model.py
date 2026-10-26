"""Checkpoint loading, greedy decoding with a KV cache, and answer extraction."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np
import torch
from experiments.config import get

LN2 = math.log(2.0)


@dataclass
class TokenPolicy:
    """Forced prefix followed by sampling up to a step boundary, then greedy decoding."""
    forced_prefix: List[int] = field(default_factory=list)
    sample_until_boundary: bool = False
    temperature: float = 1.0
    max_sample_tokens: int = 64
    boundary_fn: Optional[Callable[[str], bool]] = None
    generator: Optional[torch.Generator] = None


@dataclass
class GenOutput:
    token_ids: List[int]
    text: str
    logprob_bits: List[float]
    entropy_bits: List[float]
    h0: np.ndarray
    prompt_len: int
    finished: str
    n_sampled: int = 0
    states: Optional[np.ndarray] = None


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    dev = get(cfg, 'device', 'auto')
    if dev == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(dev)


def resolve_dtype(cfg: Dict[str, Any], device: torch.device) -> torch.dtype:
    name = str(get(cfg, 'dtype', 'bfloat16'))
    if device.type == 'cpu' and name in ('bfloat16', 'float16'):
        return torch.float32
    return {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[name]


def load_model_and_tokenizer(cfg: Dict[str, Any], model_cfg: Dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg, device)
    name, revision = model_cfg['name'], model_cfg.get('revision')
    kwargs = {'revision': revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return (model, tokenizer)


def model_dims(model, tokenizer) -> Dict[str, int]:
    cfg = model.config
    return {'d': int(cfg.hidden_size), 'v': int(getattr(cfg, 'vocab_size', len(tokenizer))), 'L': int(cfg.num_hidden_layers)}


def encode_prompt(tokenizer, prompt: str, add_bos: bool = True) -> List[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if add_bos and tokenizer.bos_token_id is not None and (not ids or ids[0] != tokenizer.bos_token_id):
        ids = [tokenizer.bos_token_id] + ids
    return ids


def sliding_window(model) -> Optional[int]:
    """Attention window of the model's local-attention layers, or None when every layer attends globally."""
    cfg = model.config
    return getattr(cfg, 'sliding_window', None) if getattr(cfg, 'cache_implementation', None) == 'hybrid' else None


def _new_cache(model, max_len: int):
    """Key-value cache for one sequence of at most max_len tokens, windowed where the model's attention is."""
    from transformers import DynamicCache, HybridCache
    if sliding_window(model):
        p = next(model.parameters())
        return HybridCache(model.config, batch_size=1, max_cache_len=max_len, device=p.device, dtype=p.dtype)
    return DynamicCache()


def _find_stop(text: str, stop_strings: Sequence[str]) -> int:
    """Index of the earliest stop string in text, or -1."""
    best = -1
    for s in stop_strings:
        j = text.find(s)
        if j >= 0 and (best < 0 or j < best):
            best = j
    return best


@torch.no_grad()
def generate(model, tokenizer, prompt_ids: List[int], max_new_tokens: int, h0_layer: int = -1,
             stop_strings: Sequence[str] = ('\nQ:',), policy: Optional[TokenPolicy] = None,
             collect_states: bool = False) -> GenOutput:
    """Decode up to max_new_tokens tokens, recording per-token surprisal and entropy, the prompt-final state and, optionally, the state at every generated position."""
    device = next(model.parameters()).device
    policy = policy or TokenPolicy()
    n_prompt = len(prompt_ids)
    window = sliding_window(model)
    past = _new_cache(model, n_prompt + max_new_tokens)
    out = model(input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=device), past_key_values=past, use_cache=True,
                output_hidden_states=True, cache_position=torch.arange(n_prompt, device=device))
    past = out.past_key_values
    logits = out.logits[0, -1].float()
    h0 = out.hidden_states[h0_layer][0, -1].float().cpu().numpy().copy()
    cur_state = h0.astype(np.float16)
    states: List[np.ndarray] = []
    eos_id = tokenizer.eos_token_id
    ids: List[int] = []
    logprobs: List[float] = []
    entropies: List[float] = []
    finished = 'cap'
    sampling = False
    n_sampled = 0
    sampled_ids: List[int] = []
    for t in range(max_new_tokens):
        if collect_states:
            states.append(cur_state)
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent = float(-(p * logp).sum().item() / LN2)
        sampled_now = False
        if t < len(policy.forced_prefix):
            nxt = int(policy.forced_prefix[t])
        elif policy.sample_until_boundary and (sampling or t == len(policy.forced_prefix)):
            sampling = True
            sampled_now = True
            probs = torch.softmax(logits / max(policy.temperature, 1e-06), dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=policy.generator).item())
            n_sampled += 1
            sampled_ids.append(nxt)
        else:
            nxt = int(torch.argmax(logits).item())
        logprobs.append(float(logp[nxt].item() / LN2))
        entropies.append(ent)
        ids.append(nxt)
        if sampled_now and nxt != eos_id:
            seg = tokenizer.decode(sampled_ids, skip_special_tokens=True)
            if policy.boundary_fn is not None and policy.boundary_fn(seg) or n_sampled >= policy.max_sample_tokens:
                sampling = False
        if eos_id is not None and nxt == eos_id:
            finished = 'eos'
            break
        if stop_strings and t >= len(policy.forced_prefix):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            if _find_stop(text, stop_strings) >= 0:
                finished = 'stop'
                break
        if t + 1 == max_new_tokens:
            break
        pos = n_prompt + t
        if window and n_prompt < window and pos == window - 1:
            past = _new_cache(model, n_prompt + max_new_tokens)
            step = model(input_ids=torch.tensor([prompt_ids + ids], dtype=torch.long, device=device), past_key_values=past,
                         use_cache=True, output_hidden_states=True, cache_position=torch.arange(pos + 1, device=device))
        else:
            step = model(input_ids=torch.tensor([[nxt]], dtype=torch.long, device=device), past_key_values=past,
                         use_cache=True, output_hidden_states=True, cache_position=torch.tensor([pos], device=device))
        past = step.past_key_values
        logits = step.logits[0, -1].float()
        if collect_states:
            cur_state = step.hidden_states[h0_layer][0, -1].float().cpu().numpy().astype(np.float16)
    if finished == 'eos' and ids and ids[-1] == eos_id:
        ids, logprobs, entropies = (ids[:-1], logprobs[:-1], entropies[:-1])
    text = tokenizer.decode(ids, skip_special_tokens=True)
    if finished == 'stop':
        j = _find_stop(text, stop_strings)
        m = len(ids)
        while m > 0 and len(tokenizer.decode(ids[:m - 1], skip_special_tokens=True)) >= j:
            m -= 1
        ids, logprobs, entropies = (ids[:m], logprobs[:m], entropies[:m])
        text = tokenizer.decode(ids, skip_special_tokens=True)[:j]
    return GenOutput(token_ids=ids, text=text, logprob_bits=logprobs, entropy_bits=entropies, h0=h0,
                     prompt_len=len(prompt_ids), finished=finished, n_sampled=n_sampled,
                     states=np.stack(states[:len(ids)]) if collect_states and states[:len(ids)] else None)


@torch.no_grad()
def hidden_at_last_position(model, ids: List[int], h0_layer: int = -1) -> np.ndarray:
    device = next(model.parameters()).device
    out = model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), output_hidden_states=True)
    return out.hidden_states[h0_layer][0, -1].float().cpu().numpy()


def extract_answer(text: str, delimiter: str = 'The answer is') -> str:
    """Text after the last occurrence of the delimiter, or '' if absent."""
    j = text.rfind(delimiter)
    if j < 0:
        return ''
    tail = text[j + len(delimiter):]
    tail = tail.split('\n')[0].strip()
    tail = tail.lstrip(':').strip()
    while tail.endswith('.'):
        tail = tail[:-1].strip()
    return tail


def split_trace_and_answer(text: str, delimiter: str = 'The answer is'):
    """(reasoning text before the last delimiter, answer text)."""
    j = text.rfind(delimiter)
    if j < 0:
        return (text, '')
    return (text[:j].rstrip(), extract_answer(text, delimiter))
