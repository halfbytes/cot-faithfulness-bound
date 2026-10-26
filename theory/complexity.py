"""Solution-trace complexity estimates."""
from __future__ import annotations
import math
import zlib
from typing import Dict, Iterable, Sequence

LOG2_26_FACTORIAL = math.log2(math.factorial(26))


def cipher_program_length(n_chars: int, bits_per_char: float = 5.0, key_bits: float = LOG2_26_FACTORIAL) -> float:
    """log2(26!) + 5 bits per letter."""
    if n_chars < 0:
        raise ValueError('n_chars must be non-negative')
    return key_bits + bits_per_char * n_chars


def modular_program_length(n_chains: int, ops_per_chain: int, bits_per_op: float = 10.0) -> float:
    if n_chains <= 0 or ops_per_chain <= 0:
        raise ValueError('chains and operations must be positive')
    return n_chains * ops_per_chain * bits_per_op


def hash_program_length(bits_per_eval: float = 13.0, expected_evals: float = 5000.0) -> float:
    return bits_per_eval * expected_evals


def multiplication_running_example() -> Dict[str, float]:
    """Bit budget of a canonical three-digit multiplication trace."""
    parts = {'single_digit_multiplications': 9 * 7.0, 'partial_product_rows': 60.0,
             'column_addition_with_carries': 50.0, 'bookkeeping': 31.0}
    parts['total'] = sum(parts.values())
    return parts


def gpqa_program_length(mean_steps: float = 4.7, bits_lo: float = 10.0, bits_hi: float = 14.0) -> Dict[str, float]:
    """Expert step count times a per-step bit range."""
    return {'low': mean_steps * bits_lo, 'high': mean_steps * bits_hi}


def lz78_phrases(text: str) -> int:
    """Number of phrases in the LZ78 incremental parsing of text."""
    dictionary = {'': 0}
    phrases = 0
    w = ''
    for ch in text:
        wc = w + ch
        if wc in dictionary:
            w = wc
        else:
            dictionary[wc] = len(dictionary)
            phrases += 1
            w = ''
    if w:
        phrases += 1
    return phrases


def lz78_bits(text: str) -> float:
    """LZ78 code length c * (log2 c + log2 alphabet) for c phrases."""
    if not text:
        return 0.0
    c = lz78_phrases(text)
    alphabet = max(1, len(set(text)))
    return c * (math.log2(max(c, 1)) + math.log2(alphabet)) if c > 0 else 0.0


def zlib_bits(text: str, level: int = 9) -> float:
    """DEFLATE compressed size in bits."""
    if not text:
        return 0.0
    return 8.0 * len(zlib.compress(text.encode('utf-8'), level))


def lz_complexity_bits(text: str, method: str = 'lz78') -> float:
    if method == 'lz78':
        return lz78_bits(text)
    if method == 'zlib':
        return zlib_bits(text)
    raise ValueError(f'unknown LZ method {method!r}')


def median(values: Sequence[float]) -> float:
    vals = sorted(values)
    if not vals:
        raise ValueError('empty')
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def relative_agreement(a: float, b: float) -> float:
    """|a-b| / max(a,b)."""
    if max(a, b) == 0:
        return 0.0
    return abs(a - b) / max(a, b)


def gpqa_expert_trace_bits(traces: Iterable[str], method: str = 'lz78', n_traces: int = 50) -> Dict[str, float]:
    """LZ code length of expert-written solution traces."""
    vals = [lz_complexity_bits(t, method=method) for t in list(traces)[:n_traces] if t and t.strip()]
    if not vals:
        return {'n': 0, 'median_bits': 0.0, 'min_bits': 0.0, 'max_bits': 0.0}
    return {'n': len(vals), 'median_bits': median(vals), 'min_bits': min(vals), 'max_bits': max(vals),
            'mean_bits': sum(vals) / len(vals)}


def gpqa_k_tilde_range(program_bits: Dict[str, float], expert_bits: Dict[str, float]) -> Dict[str, float]:
    """Range from the program-length lower figure to the expert-trace median."""
    lo = program_bits['low']
    hi = expert_bits['median_bits']
    return {'low': lo, 'high': hi, 'relative_disagreement': relative_agreement(program_bits['high'], hi)}
