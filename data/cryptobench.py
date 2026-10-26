"""CryptoBench generator, answer checking, complexity report and the released-file format."""
from __future__ import annotations
import json
import math
import random
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple
from theory import complexity
from data.schema import Benchmark, Exemplar, Instance, strip_answer

ALPHABET = string.ascii_uppercase
CHAIN_NAMES = list('ABCDEFGHIJ')
HASH_MODULUS = 10007
HASH_MODULUS_2 = 8191
HASH_SPACE = 10000
WORDS = 'the a an and or but if then of to in on at by for with from into over under about after before between through during without within along across behind beyond near cat dog fox owl bee ant cow pig hen ram elk yak bat rat fish bird wolf bear deer goat lion tiger horse mouse sun moon star sky sea lake river hill road path tree leaf root seed rock sand snow rain wind fire ice mist red blue green gold gray black white brown pink dark pale bright small large tall short old new warm cold quick slow quiet loud brave calm kind wise sad glad king queen child friend farmer sailor doctor baker miner poet judge guard house barn tent boat ship cart wheel door gate wall roof bridge tower garden field forest village city market runs walks jumps sings reads writes builds finds keeps gives takes holds sees hears wants needs makes eats sleeps waits hides opens closes climbs falls swims flies draws paints book map key coin ring bell drum song tale plan gift bread milk honey apple pear corn rice salt water wine morning evening night summer winter spring autumn week year time day hour one two three four five six seven eight nine ten many few some all each every'.split()


def _random_plaintext(rng: random.Random, min_len: int, max_len: int) -> str:
    target = rng.randint(min_len, max_len)
    words: List[str] = []
    while True:
        cand = ' '.join(words + [rng.choice(WORDS)])
        if len(cand) > max_len:
            break
        words.append(cand.split()[-1])
        if len(cand) >= target:
            break
    text = ' '.join(words).upper()
    if len(text) < min_len:
        while len(text) < min_len:
            w = rng.choice(WORDS).upper()
            if len(text) + 1 + len(w) <= max_len:
                text = f'{text} {w}'
            else:
                break
    return text


def apply_substitution(text: str, key: Dict[str, str]) -> str:
    return ''.join(key[ch] if ch in key else ch for ch in text)


def make_cipher_problem(rng: random.Random, idx: int, min_len: int = 40, max_len: int = 100, n_revealed: int = 3) -> Instance:
    plaintext = _random_plaintext(rng, min_len, max_len)
    perm = list(ALPHABET)
    rng.shuffle(perm)
    key = dict(zip(ALPHABET, perm))
    ciphertext = apply_substitution(plaintext, key)
    letters = [c for c in ciphertext if c.isalpha()]
    n = len(letters)
    freq = Counter(letters).most_common()
    top = ', '.join(f'{c} ({k})' for c, k in freq[:6])
    plain_freq = Counter(c for c in plaintext if c.isalpha()).most_common(n_revealed)
    revealed = ', '.join(f'{key[p]} -> {p}' for p, _ in plain_freq)
    question = (f'The following message was encrypted with a random simple substitution cipher on the 26 uppercase letters '
                f'(spaces were left unchanged).\nCiphertext: {ciphertext}\nFrequency-analysis hints: the most frequent ciphertext '
                f'letters are {top}; the most frequent letters in English are E, T, A, O, I, N, S, H, R; and the following '
                f'cipher-to-plain mappings are known: {revealed}.\nRecover the plaintext.')
    inv = {v: k for k, v in key.items()}
    used = sorted(set(letters))
    key_line = ' '.join(f'{c}->{inv[c]}' for c in used)
    verbose = (f'Using the frequency hints and the known mappings, the cipher-to-plain key for the letters that occur is: '
               f'{key_line}. Substituting letter by letter gives {plaintext}. The answer is {plaintext}.')
    canonical = ''.join(key[a] for a in ALPHABET) + '|' + plaintext
    return Instance(id=f'cipher-{idx:04d}', question=question, answer=plaintext, trace=verbose,
                    meta={'category': 'cipher', 'n_letters': n, 'ciphertext': ciphertext, 'key': key, 'canonical_trace': canonical,
                          'K_program_bits': complexity.cipher_program_length(n)})


def normalize_cipher_answer(raw: str) -> str:
    s = strip_answer(raw).upper()
    s = ''.join(ch if ch.isalpha() or ch == ' ' else ' ' for ch in s)
    return ' '.join(s.split())


VOCABULARY = frozenset(w.upper() for w in WORDS)


def is_consistent_decryption(plaintext: str, question: str) -> bool:
    """Whether plaintext uses only generator words and one letter substitution maps it onto the ciphertext with the revealed mappings."""
    m = re.search('Ciphertext: (.*)\\n.*mappings are known: (.*)\\.\\n', question)
    if not m:
        return False
    cwords, pwords = (m.group(1).split(' '), plaintext.split(' '))
    if len(cwords) != len(pwords) or not all(w in VOCABULARY for w in pwords):
        return False
    fwd: Dict[str, str] = {}
    inv: Dict[str, str] = {}
    for cw, pw in zip(cwords, pwords):
        if len(cw) != len(pw):
            return False
        for c, p in zip(cw, pw):
            if fwd.setdefault(p, c) != c or inv.setdefault(c, p) != p:
                return False
    return all(inv.get(c) == p for c, p in (pair.split(' -> ') for pair in m.group(2).split(', ')))


def _primes_between(lo: int, hi: int) -> List[int]:
    sieve = [True] * (hi + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(hi ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, hi + 1, i):
                sieve[j] = False
    return [p for p in range(lo, hi + 1) if sieve[p]]


PRIMES_101_997 = _primes_between(101, 997)


def make_modular_problem(rng: random.Random, idx: int, min_chains: int = 8, max_chains: int = 10, min_ops: int = 8,
                         max_ops: int = 15, cross_ref_prob: float = 0.25) -> Instance:
    p = rng.choice(PRIMES_101_997)
    n_chains = rng.randint(min_chains, max_chains)
    names = CHAIN_NAMES[:n_chains]
    n_ops = {c: rng.randint(min_ops, max_ops) for c in names}
    values = {c: rng.randint(0, p - 1) for c in names}
    starts = dict(values)
    remaining = dict(n_ops)
    step_lines: List[str] = []
    verbose_steps: List[str] = []
    canonical_vals: List[str] = []
    step = 0
    while any(remaining.values()):
        for c in names:
            if remaining[c] <= 0:
                continue
            step += 1
            remaining[c] -= 1
            op = rng.choice(['+', '-', '*'])
            if n_chains > 1 and rng.random() < cross_ref_prob:
                other = rng.choice([o for o in names if o != c])
                operand_desc = f'the current value of {other}'
                operand = values[other]
            else:
                operand = rng.randint(1, p - 1)
                operand_desc = str(operand)
            before = values[c]
            if op == '+':
                after = (before + operand) % p
            elif op == '-':
                after = (before - operand) % p
            else:
                after = before * operand % p
            values[c] = after
            step_lines.append(f'Step {step}: {c} = ({c} {op} {operand_desc}) mod {p}.')
            verbose_steps.append(f'Step {step}: {c} = ({before} {op} {operand}) mod {p} = {after}.')
            canonical_vals.append(f'{after:03d}')
    start_desc = ', '.join(f'{c} starts at {starts[c]}' for c in names)
    chain_list = ', '.join(names)
    form = ', '.join(f'{c}=value' for c in names)
    question = (f'Work modulo the prime p = {p}. There are {n_chains} chains named {chain_list}; {start_desc}. Apply the following '
                f'operations in order (a reference to another chain uses its current value at that step):\n'
                + '\n'.join(step_lines) + '\nWhat are the final values of all chains? Give them in the form ' + form + '.')
    answer = ', '.join(f'{c}={values[c]}' for c in names)
    verbose = ' '.join(verbose_steps) + f' The final values are {answer}. The answer is {answer}.'
    total_ops = sum(n_ops.values())
    return Instance(id=f'modular-{idx:04d}', question=question, answer=answer, trace=verbose,
                    meta={'category': 'modular', 'p': p, 'n_chains': n_chains, 'ops_per_chain': n_ops, 'total_ops': total_ops,
                          'canonical_trace': ''.join(canonical_vals),
                          'K_program_bits': complexity.modular_program_length(1, total_ops, bits_per_op=10.0),
                          'K_program_bits_exact_prime': complexity.modular_program_length(1, total_ops, bits_per_op=math.log2(p))})


def parse_modular_answer(raw: str) -> Dict[str, int]:
    """Final value per chain name; values given without names are assigned to A, B, C, ... in order."""
    s = strip_answer(raw)
    pairs = re.findall('([A-J])\\s*=\\s*(-?\\d+)', s.upper())
    if pairs:
        out: Dict[str, int] = {}
        for c, v in pairs:
            if out.setdefault(c, int(v)) != int(v):
                return {}
        return out
    return dict(zip(CHAIN_NAMES, (int(v) for v in re.findall('-?\\d+', s))))


def hash_value(n: int, a: int, b: int, c: int, d: int) -> int:
    return (((a * n + b) % HASH_MODULUS) ^ ((c * n * n + d) % HASH_MODULUS_2)) % HASH_SPACE


def make_hash_problem(rng: random.Random, idx: int) -> Instance:
    a = rng.randint(1, HASH_MODULUS - 1)
    b = rng.randint(0, HASH_MODULUS - 1)
    c = rng.randint(1, HASH_MODULUS_2 - 1)
    d = rng.randint(0, HASH_MODULUS_2 - 1)
    n_star = rng.randint(0, HASH_SPACE - 1)
    target = hash_value(n_star, a, b, c, d)
    evals: List[Tuple[int, int]] = []
    canon_vals: List[str] = []
    first = None
    for n in range(HASH_SPACE):
        h = hash_value(n, a, b, c, d)
        evals.append((n, h))
        canon_vals.append(f'{h:04d}')
        if h == target:
            first = n
            break
    assert first is not None
    question = (f'Define the hash function h(n) = ((({a} * n + {b}) mod {HASH_MODULUS}) XOR (({c} * n^2 + {d}) mod {HASH_MODULUS_2})) '
                f'mod {HASH_SPACE}, where XOR is the bitwise exclusive or. It maps every integer n with 0 <= n <= {HASH_SPACE - 1} '
                f'to a four-digit value. Find an integer n in that range with h(n) = {target:04d}.')
    tail = ', '.join(f'h({n}) = {h:04d}' for n, h in evals[-5:])
    verbose = (f'Evaluating h(n) for n = 0, 1, 2, ... and comparing with {target:04d}: {tail}. The first match is n = {first}. '
               f'The answer is {first}.')
    return Instance(id=f'hash-{idx:04d}', question=question, answer=str(first), trace=verbose,
                    meta={'category': 'hash', 'a': a, 'b': b, 'c': c, 'd': d, 'target': target, 'n_star': n_star, 'n_evaluations': len(evals),
                          'canonical_trace': ''.join(canon_vals), 'K_program_bits': complexity.hash_program_length(),
                          'K_program_bits_actual': 13.0 * len(evals)})


def check_hash_answer(raw: str, inst: Instance) -> bool:
    m = re.search('-?\\d+', strip_answer(raw))
    if not m:
        return False
    n = int(m.group(0))
    if not 0 <= n < HASH_SPACE:
        return False
    meta = inst.meta
    return hash_value(n, meta['a'], meta['b'], meta['c'], meta['d']) == meta['target']


def generate_cryptobench(seed: int = 0, n_cipher: int = 200, n_modular: int = 200, n_hash: int = 100,
                         cipher_kwargs: Optional[Dict] = None, modular_kwargs: Optional[Dict] = None) -> List[Instance]:
    rng = random.Random(seed)
    out: List[Instance] = []
    for i in range(n_cipher):
        out.append(make_cipher_problem(rng, i, **cipher_kwargs or {}))
    for i in range(n_modular):
        out.append(make_modular_problem(rng, i, **modular_kwargs or {}))
    for i in range(n_hash):
        out.append(make_hash_problem(rng, i))
    return out


def release_record(inst: Instance) -> Dict:
    """Fields of one instance as stored in the released instance file."""
    m = inst.meta
    d = {'id': inst.id, 'category': m['category'], 'question': inst.question, 'answer': inst.answer, 'K_program_bits': m['K_program_bits']}
    if m['category'] == 'cipher':
        d['n_letters'] = m['n_letters']
    elif m['category'] == 'modular':
        d.update({'p': m['p'], 'n_chains': m['n_chains'], 'total_ops': m['total_ops']})
    else:
        d.update({'target': m['target'], 'n_evaluations': m['n_evaluations']})
    return d


def release_payload(instances: Sequence[Instance]) -> str:
    """Uncompressed content of the released instance file."""
    return '\n'.join(json.dumps(release_record(i)) for i in instances) + '\n'


def complexity_report(instances: Sequence[Instance], lz_method: str = 'lz78') -> Dict[str, Dict[str, float]]:
    """Program-length and LZ estimates per category, per instance and pooled."""
    report: Dict[str, Dict[str, float]] = {}
    by_cat: Dict[str, List[Instance]] = {}
    for inst in instances:
        by_cat.setdefault(inst.meta['category'], []).append(inst)
    for cat, insts in by_cat.items():
        prog = [float(i.meta['K_program_bits']) for i in insts]
        lz = [complexity.lz_complexity_bits(i.meta['canonical_trace'], method=lz_method) for i in insts]
        med_p, med_l = (complexity.median(prog), complexity.median(lz))
        joint_bits = complexity.lz_complexity_bits('\n'.join(i.meta['canonical_trace'] for i in insts), method=lz_method)
        joint_per_instance = joint_bits / len(insts)
        mean_p = sum(prog) / len(prog)
        report[cat] = {'n': len(insts), 'program_bits_median': med_p, 'program_bits_min': min(prog), 'program_bits_max': max(prog),
                       'program_bits_mean': mean_p, 'lz_bits_median': med_l,
                       'relative_disagreement': complexity.relative_agreement(med_p, med_l),
                       'lz_bits_joint_total': joint_bits, 'lz_bits_joint_per_instance': joint_per_instance,
                       'relative_disagreement_joint': complexity.relative_agreement(mean_p, joint_per_instance)}
        if cat == 'modular':
            report[cat]['program_bits_exact_prime_median'] = complexity.median([float(i.meta['K_program_bits_exact_prime']) for i in insts])
        if cat == 'cipher':
            report[cat]['n_letters_median'] = complexity.median([float(i.meta['n_letters']) for i in insts])
    return report


class CryptoBench(Benchmark):
    name = 'cryptobench'

    def normalize(self, raw: str) -> str:
        return strip_answer(raw)

    def is_correct(self, pred_raw: str, inst: Instance) -> bool:
        cat = inst.meta.get('category')
        if cat == 'cipher':
            pred = normalize_cipher_answer(pred_raw)
            return pred == normalize_cipher_answer(inst.answer) or is_consistent_decryption(pred, inst.question)
        if cat == 'modular':
            return parse_modular_answer(pred_raw) == parse_modular_answer(inst.answer)
        if cat == 'hash':
            return check_hash_answer(pred_raw, inst)
        return super().is_correct(pred_raw, inst)


def exemplars_from_generator(seed: int, n_cipher: int = 3, n_modular: int = 3, n_hash: int = 2, short: bool = True) -> List[Exemplar]:
    """Few-shot exemplars generated from a separate seed, interleaved across the three categories."""
    ck = {'min_len': 40, 'max_len': 60} if short else {}
    mk = {'min_chains': 2, 'max_chains': 3, 'min_ops': 3, 'max_ops': 5} if short else {}
    insts = generate_cryptobench(seed=seed, n_cipher=n_cipher, n_modular=n_modular, n_hash=n_hash, cipher_kwargs=ck, modular_kwargs=mk)
    by_cat: Dict[str, List[Instance]] = {}
    for i in insts:
        by_cat.setdefault(i.meta['category'], []).append(i)
    order: List[Instance] = []
    while any(by_cat.values()):
        for cat in ('cipher', 'modular', 'hash'):
            if by_cat.get(cat):
                order.append(by_cat[cat].pop(0))
    out: List[Exemplar] = []
    for inst in order:
        rationale = inst.trace
        marker = ' The answer is '
        if marker in rationale:
            rationale = rationale[:rationale.rfind(marker)]
        out.append(Exemplar(question=inst.question, rationale=rationale.strip(), answer=inst.answer))
    return out
