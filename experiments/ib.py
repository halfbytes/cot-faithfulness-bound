"""Bottleneck-penalised LoRA fine-tuning and the accuracy/sensitivity sweep over the penalty weight."""
from __future__ import annotations
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from experiments.config import get
from data.prompts import build_prompt, completion_text
from data.registry import generation_cap, get_benchmark
from experiments.model import load_model_and_tokenizer
from experiments.seeding import seed_everything

LN2 = math.log(2.0)


class GaussianBottleneck(nn.Module):
    """Gaussian encoder with a factorised prior; returns the perturbed states and records the KL in bits per position, averaged or summed over hidden dimensions."""

    def __init__(self, d: int, logsigma_init: float = -6.0, learnable_prior: bool = True, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError(f'unknown KL reduction {reduction!r}')
        self.reduction = reduction
        self.mu = nn.Linear(d, d)
        self.logsigma = nn.Linear(d, d)
        nn.init.eye_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logsigma.weight)
        nn.init.constant_(self.logsigma.bias, logsigma_init)
        self.prior_mu = nn.Parameter(torch.zeros(d), requires_grad=learnable_prior)
        self.prior_logsigma = nn.Parameter(torch.zeros(d), requires_grad=learnable_prior)
        self.enabled = True
        self.mask: Optional[torch.Tensor] = None
        self.last_kl_bits: Optional[torch.Tensor] = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.enabled or (self.mask is None and h.shape[1] > 1):
            return h
        dtype = h.dtype
        hf = h.float()
        mu = self.mu(hf)
        logsigma = self.logsigma(hf).clamp(-10.0, 3.0)
        std = logsigma.exp()
        eps = torch.randn_like(std) if self.training else torch.zeros_like(std)
        z = mu + std * eps
        pm = self.prior_mu.view(1, 1, -1)
        pls = self.prior_logsigma.view(1, 1, -1).clamp(-10.0, 3.0)
        kl_nats = (pls - logsigma + (std.pow(2) + (mu - pm).pow(2)) / (2 * (2 * pls).exp()) - 0.5).sum(-1)
        if self.reduction == 'mean':
            kl_nats = kl_nats / h.shape[-1]
        if self.mask is not None:
            m = self.mask.to(kl_nats.dtype)
            denom = m.sum().clamp_min(1.0)
            self.last_kl_bits = (kl_nats * m).sum() / denom / LN2
            z = torch.where(self.mask.unsqueeze(-1).bool(), z, hf)
        else:
            self.last_kl_bits = kl_nats.mean() / LN2
        return z.to(dtype)

    def set_generation_mode(self) -> None:
        """Act on each decoded position and leave the prompt pass unchanged, as at the training positions."""
        self.mask = None


def _attach(model, bottleneck: GaussianBottleneck):
    """Insert the bottleneck between the final decoder norm and the LM head."""
    base = model.get_decoder() if hasattr(model, 'get_decoder') else model.model
    norm = base.norm
    if getattr(norm, '_ib_wrapped', False):
        norm._ib_bottleneck = bottleneck
        return
    orig_forward = norm.forward

    def wrapped(hidden_states, *args, **kwargs):
        out = orig_forward(hidden_states, *args, **kwargs)
        bn = getattr(norm, '_ib_bottleneck', None)
        return bn(out) if bn is not None else out
    norm.forward = wrapped
    norm._ib_wrapped = True
    norm._ib_bottleneck = bottleneck


def build_training_examples(cfg: Dict[str, Any], bench_name: str, tokenizer, max_examples: Optional[int] = None):
    """(prompt, completion) pairs from the training split; the completion is the reference trace and answer."""
    benchmark = get_benchmark(cfg, bench_name)
    train_insts = benchmark.load('train')
    train_ids = {i.id for i in train_insts}
    for split in ('eval', 'validation'):
        try:
            other_ids = {i.id for i in benchmark.load(split)}
        except FileNotFoundError:
            continue
        overlap = len(train_ids & other_ids)
        if overlap:
            raise RuntimeError(f'{bench_name}: train and {split} splits share {overlap} instances')
    k = int(get(cfg, 'ib.train_k_shot', 8))
    delimiter = str(get(cfg, 'prompting.delimiter', 'The answer is'))
    exemplars = benchmark.exemplars(k)
    out: List[Tuple[str, str]] = []
    for inst in train_insts:
        trace = inst.trace
        if not trace:
            continue
        body = trace[:trace.rfind(delimiter)].strip() if delimiter in trace else trace.strip()
        out.append((build_prompt(inst.question, exemplars, delimiter), completion_text(body, inst.answer, delimiter)))
        if max_examples and len(out) >= max_examples:
            break
    return out


def _encode_example(tokenizer, prompt: str, completion: str, max_len: int):
    """(token ids, labels, completion mask, whether the completion was cut at max_len)."""
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    c_ids = tokenizer.encode(completion, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        p_ids = [tokenizer.bos_token_id] + p_ids
    if tokenizer.eos_token_id is not None:
        c_ids = c_ids + [tokenizer.eos_token_id]
    ids = (p_ids + c_ids)[:max_len]
    n_prompt = min(len(p_ids), len(ids))
    labels = [-100] * n_prompt + ids[n_prompt:]
    trace_mask = [0] * n_prompt + [1] * (len(ids) - n_prompt)
    return (ids, labels, trace_mask, len(p_ids) + len(c_ids) > max_len)


def train_ib(cfg: Dict[str, Any], model_key: str, bench_name: str, beta: float, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Fine-tune with LoRA, adding the bottleneck penalty when beta > 0; save the adapter and bottleneck."""
    from peft import LoraConfig, get_peft_model
    seed = int(get(cfg, 'ib.seed', 0))
    seed_everything(seed)
    model, tokenizer = load_model_and_tokenizer(cfg, cfg['models'][model_key])
    device = next(model.parameters()).device
    if bool(get(cfg, 'ib.gradient_checkpointing', True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    lora = LoraConfig(r=int(get(cfg, 'ib.lora_r', 16)), lora_alpha=int(get(cfg, 'ib.lora_alpha', 32)),
                      lora_dropout=float(get(cfg, 'ib.lora_dropout', 0.0)),
                      target_modules=list(get(cfg, 'ib.target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])),
                      task_type='CAUSAL_LM', bias='none')
    model = get_peft_model(model, lora)
    d = model.config.hidden_size
    bottleneck = GaussianBottleneck(d, float(get(cfg, 'ib.logsigma_init', -6.0)),
                                    str(get(cfg, 'ib.prior', 'learnable')) == 'learnable',
                                    str(get(cfg, 'ib.kl_reduction', 'mean'))).to(device)
    bottleneck.enabled = beta > 0
    _attach(model.base_model.model if hasattr(model, 'base_model') else model, bottleneck)
    seen, params = (set(), [])
    for p_ in list(model.parameters()) + list(bottleneck.parameters()):
        if p_.requires_grad and id(p_) not in seen:
            seen.add(id(p_))
            params.append(p_)
    opt = torch.optim.AdamW(params, lr=float(get(cfg, 'ib.lr', 2e-05)), weight_decay=float(get(cfg, 'ib.weight_decay', 0.0)))
    examples = build_training_examples(cfg, bench_name, tokenizer, get(cfg, 'ib.max_train_examples'))
    if not examples:
        raise RuntimeError(f'no training examples with reference traces for {bench_name}')
    max_len = int(get(cfg, 'ib.max_seq_len', 8192))
    encoded = [_encode_example(tokenizer, p, c, max_len) for p, c in examples]
    usable = [e for e in encoded if any(e[2])]
    n_skipped, n_truncated = (len(encoded) - len(usable), sum(1 for e in usable if e[3]))
    if not usable:
        raise RuntimeError(f'{bench_name}: no training example leaves room for its completion within ib.max_seq_len={max_len}')
    if n_skipped:
        print(f'[ib] {bench_name}: {n_skipped} of {len(encoded)} examples have no completion token within ib.max_seq_len={max_len}; skipped')
    epochs = int(get(cfg, 'ib.epochs', 3))
    batch_size = int(get(cfg, 'ib.batch_size', 16))
    micro = int(get(cfg, 'ib.micro_batch_size', 2))
    accum = max(1, batch_size // micro)
    if len(usable) < micro * accum:
        raise RuntimeError(f'{bench_name}: {len(usable)} usable training examples, fewer than one batch of {micro * accum}')
    rng = np.random.default_rng(seed)
    model.train()
    bottleneck.train()
    history: List[Dict[str, float]] = []
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(usable))
        n_micro = (len(order) // (micro * accum)) * accum
        for start in range(0, n_micro * micro, micro):
            enc = [usable[i] for i in order[start:start + micro]]
            L = max(len(e[0]) for e in enc)
            pad = tokenizer.pad_token_id or 0
            ids = torch.tensor([e[0] + [pad] * (L - len(e[0])) for e in enc], device=device)
            labels = torch.tensor([e[1] + [-100] * (L - len(e[1])) for e in enc], device=device)
            attn = torch.tensor([[1] * len(e[0]) + [0] * (L - len(e[0])) for e in enc], device=device)
            tmask = torch.tensor([e[2] + [0] * (L - len(e[2])) for e in enc], device=device)
            bottleneck.mask = tmask
            out = model(input_ids=ids, attention_mask=attn, labels=labels, use_cache=False)
            lm_loss = out.loss
            kl_bits = bottleneck.last_kl_bits if beta > 0 and bottleneck.last_kl_bits is not None else torch.zeros((), device=device)
            loss = lm_loss + beta * kl_bits
            (loss / accum).backward()
            step += 1
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], float(get(cfg, 'ib.grad_clip', 1.0)))
                opt.step()
                opt.zero_grad(set_to_none=True)
            history.append({'epoch': epoch, 'step': step, 'lm_loss': float(lm_loss.item()),
                            'kl_bits': float(kl_bits.item()) if torch.is_tensor(kl_bits) else 0.0, 'loss': float(loss.item())})
        recent = np.mean([h['lm_loss'] for h in history[-50:]])
        print(f'[ib] {bench_name} beta={beta} epoch={epoch} lm_loss={recent:.4f}')
    out_dir = out_dir or os.path.join(get(cfg, 'paths.checkpoints', 'checkpoints'), 'ib', model_key, bench_name, f'beta{beta}')
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    torch.save(bottleneck.state_dict(), os.path.join(out_dir, 'bottleneck.pt'))
    tail = [h['loss'] for h in history[-500:]]
    info = {'model': model_key, 'benchmark': bench_name, 'beta': beta, 'epochs': epochs, 'n_examples': len(usable),
            'n_examples_skipped': n_skipped, 'n_examples_truncated': n_truncated,
            'n_steps': step, 'final_loss_mean': float(np.mean(tail)) if tail else 0.0,
            'final_loss_std': float(np.std(tail)) if tail else 0.0, 'path': out_dir, 'seed': seed}
    with open(os.path.join(out_dir, 'train_info.json'), 'w', encoding='utf-8') as f:
        json.dump({**info, 'history': history[-200:]}, f, indent=2)
    return info


def load_ib_model(cfg: Dict[str, Any], model_key: str, adapter_dir: str, beta: float):
    """Base model with the trained adapter; the bottleneck is attached when it was active in training."""
    from peft import PeftModel
    model, tokenizer = load_model_and_tokenizer(cfg, cfg['models'][model_key])
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    bpath = os.path.join(adapter_dir, 'bottleneck.pt')
    if beta > 0 and os.path.exists(bpath):
        bottleneck = GaussianBottleneck(model.config.hidden_size).to(next(model.parameters()).device)
        bottleneck.load_state_dict(torch.load(bpath, map_location='cpu', weights_only=True))
        bottleneck.eval()
        _attach(model.base_model.model if hasattr(model, 'base_model') else model, bottleneck)
        return (model, tokenizer, bottleneck)
    return (model, tokenizer, None)


def run_frontier(cfg: Dict[str, Any], model_key: str, bench_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Accuracy and sensitivity of the base model and of each fine-tuned model in the penalty sweep."""
    from experiments.generation import run_generation
    from experiments.perturbation import run_perturbation
    bench_names = bench_names or list(get(cfg, 'ib.benchmarks', ['gsm8k', 'cryptobench']))
    betas = [float(b) for b in get(cfg, 'ib.betas', [0.0, 0.01, 0.1, 1.0, 10.0])]
    results: Dict[str, List[Dict[str, Any]]] = {}
    for bench in bench_names:
        cap = generation_cap(cfg, bench)
        points = []
        base_gen_dir = os.path.join(get(cfg, 'paths.results', 'results'), 'ib', model_key, bench, 'base')
        base = load_model_and_tokenizer(cfg, cfg['models'][model_key])
        gs = run_generation(cfg, model_key, bench, 'eval', cap=cap, model_and_tok=base, out_dir=base_gen_dir, tag='_ibbase')
        ps = run_perturbation(cfg, model_key, bench, cap=cap, model_and_tok=base, gen_dir=base_gen_dir,
                              out_dir=os.path.join(base_gen_dir, 'ps'))
        del base
        points.append({'beta': None, 'label': 'base', 'accuracy': 100 * gs['accuracy'], 'PS': ps['PS_mean']})
        for beta in betas:
            info = train_ib(cfg, model_key, bench, beta)
            model, tokenizer, bn = load_ib_model(cfg, model_key, info['path'], beta)
            gdir = os.path.join(get(cfg, 'paths.results', 'results'), 'ib', model_key, bench, f'beta{beta}')
            gs = run_generation(cfg, model_key, bench, 'eval', cap=cap, model_and_tok=(model, tokenizer), out_dir=gdir,
                                tag=f'_ib{beta}', bottleneck=bn)
            ps = run_perturbation(cfg, model_key, bench, cap=cap, model_and_tok=(model, tokenizer), gen_dir=gdir,
                                  out_dir=os.path.join(gdir, 'ps'))
            label = 'lora_only' if beta == 0.0 else f'beta={beta}'
            points.append({'beta': beta, 'label': label, 'accuracy': 100 * gs['accuracy'], 'PS': ps['PS_mean'],
                           'final_loss_std': info['final_loss_std'], 'n_steps': info['n_steps']})
            print(f"[frontier] {bench} beta={beta} acc={100 * gs['accuracy']:.1f}% PS={ps['PS_mean']:.3f}")
            del model
        results[bench] = points
    d = os.path.join(get(cfg, 'paths.results', 'results'), 'ib', model_key)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'frontier.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return results
