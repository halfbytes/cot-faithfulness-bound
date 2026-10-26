"""Instance records, JSONL I/O, answer normalisation and the benchmark base class."""
from __future__ import annotations
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Instance:
    id: str
    question: str
    answer: str
    trace: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Instance':
        return Instance(id=str(d['id']), question=d['question'], answer=str(d['answer']), trace=d.get('trace'),
                        meta=d.get('meta', {}) or {})


@dataclass
class Exemplar:
    question: str
    rationale: str
    answer: str


def write_jsonl(path: str, instances: Iterable[Instance]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, 'w', encoding='utf-8') as f:
        for inst in instances:
            f.write(inst.to_json() + '\n')
            n += 1
    return n


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Instance]:
    out: List[Instance] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Instance.from_dict(json.loads(line)))
            if limit is not None and len(out) >= limit:
                break
    return out


_NUM_RE = re.compile('-?\\d[\\d,]*\\.?\\d*(?:[eE][-+]?\\d+)?|-?\\.\\d+')


def strip_answer(raw: str) -> str:
    """First line of the extracted answer without trailing periods or wrapping quotes."""
    s = raw.strip()
    s = s.split('\n')[0].strip()
    s = s.lstrip(':').strip()
    while s.endswith('.'):
        s = s[:-1].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in '"\'':
        s = s[1:-1]
    return s.strip()


def parse_number(raw: str) -> Optional[float]:
    s = strip_answer(raw).replace('$', '').replace(',', '')
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def normalize_numeric(raw: str) -> str:
    v = parse_number(raw)
    if v is None:
        return strip_answer(raw).lower()
    if abs(v - round(v)) < 1e-09:
        return str(int(round(v)))
    return f'{v:.6g}'


_MC_RE = re.compile('\\(?\\b([A-Ha-h])\\b\\)?')


def normalize_choice(raw: str, letters: str = 'ABCD') -> str:
    """Multiple-choice letter from forms such as 'B', '(B)', 'B.' or 'option B'."""
    s = strip_answer(raw)
    m = re.match('^\\(?([A-Za-z])\\)?(?:[\\s.:,]|$)', s)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    m = _MC_RE.search(s)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    return s.upper()


def normalize_yes_no(raw: str) -> str:
    s = strip_answer(raw).lower()
    s = re.sub('[^a-z]', ' ', s).strip()
    first = s.split(' ')[0] if s else ''
    if first in ('yes', 'true', 'y'):
        return 'yes'
    if first in ('no', 'false', 'n'):
        return 'no'
    return s


def normalize_boardgame(raw: str) -> str:
    s = strip_answer(raw).lower()
    for lab in ('disproved', 'proved', 'unknown'):
        if lab in s:
            return lab
    return s


def normalize_math(raw: str) -> str:
    """Canonical string form of a LaTeX answer."""
    s = raw.strip()
    m = re.search('\\\\boxed\\s*{', s)
    if m:
        s = _extract_braced(s, m.end() - 1)
    s = s.split('\n')[0].strip()
    while s.endswith('.'):
        s = s[:-1]
    s = s.replace('\\!', '').replace('\\,', '').replace('\\;', '').replace('\\ ', '')
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    s = s.replace('^{\\circ}', '').replace('^\\circ', '').replace('\\%', '').replace('%', '')
    s = s.replace('\\$', '').replace('$', '')
    s = re.sub('\\\\text\\{\\s*([^}]*)\\}', '\\1', s)
    s = re.sub('\\\\mbox\\{\\s*([^}]*)\\}', '\\1', s)
    s = s.replace(' ', '')
    s = re.sub('^[a-zA-Z]=', '', s)
    if s.startswith('(') and s.endswith(')') and ',' not in s:
        s = s[1:-1]
    s = _simple_frac(s)
    return s


def _extract_braced(s: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return s[open_idx + 1:i]
    return s[open_idx + 1:]


def _simple_frac(s: str) -> str:
    """Numeric fractions in lowest terms."""
    m = re.fullmatch('\\\\frac\\{(-?\\d+)\\}\\{(-?\\d+)\\}', s)
    if m:
        s = f'{m.group(1)}/{m.group(2)}'
    m = re.fullmatch('(-?\\d+)/(-?\\d+)', s)
    if m:
        from math import gcd
        a, b = (int(m.group(1)), int(m.group(2)))
        if b != 0:
            g = gcd(abs(a), abs(b)) or 1
            a, b = (a // g, b // g)
            if b < 0:
                a, b = (-a, -b)
            return str(a) if b == 1 else f'{a}/{b}'
    return s


class Benchmark:
    """Name, answer space, exemplars, normalisation and correctness of one benchmark."""
    name: str = 'base'
    answer_space_size: Optional[int] = None
    chance_accuracy: float = 0.0

    def __init__(self, data_dir: str, exemplars: List[Exemplar]):
        self.data_dir = data_dir
        self._exemplars = exemplars

    def path(self, split: str) -> str:
        return os.path.join(self.data_dir, f'{split}.jsonl')

    def load(self, split: str = 'eval', limit: Optional[int] = None) -> List[Instance]:
        p = self.path(split)
        if not os.path.exists(p):
            raise FileNotFoundError(f'{self.name}: missing {p}. Run `python main.py prepare-data` or `python main.py make-cryptobench`.')
        return read_jsonl(p, limit=limit)

    def exemplars(self, k: int) -> List[Exemplar]:
        if k > len(self._exemplars):
            raise ValueError(f'{self.name}: only {len(self._exemplars)} exemplars available, requested k={k}')
        return self._exemplars[:k]

    def normalize(self, raw: str) -> str:
        return strip_answer(raw).lower()

    def is_correct(self, pred_raw: str, inst: Instance) -> bool:
        return self.normalize(pred_raw) == self.normalize(inst.answer)
