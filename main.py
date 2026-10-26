"""Entry point for data preparation, verification, every measurement and aggregation."""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.registry import ALL_BENCHMARKS, generation_cap, make_cryptobench, prepare_public
from experiments.config import load_config


def _cfg(args):
    return load_config(args.config, overrides=args.set or [])


def _benchmarks(cfg, args):
    return getattr(args, 'benchmarks', None) or list(cfg.get('benchmarks') or ALL_BENCHMARKS)


def _models(cfg, args):
    if getattr(args, 'models', None):
        return args.models
    if cfg.get('model'):
        return [cfg['model']]
    return list((cfg.get('models') or {}).keys())


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_prepare_data(args) -> int:
    cfg = _cfg(args)
    _print(prepare_public(cfg, _benchmarks(cfg, args), token=args.hf_token or os.environ.get('HF_TOKEN')))
    return 0


def cmd_make_cryptobench(args) -> int:
    _print(make_cryptobench(_cfg(args)))
    return 0


def cmd_verify_cryptobench(args) -> int:
    import gzip
    import hashlib
    from data.cryptobench import generate_cryptobench, release_payload

    root = args.release_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cryptobench')
    with open(os.path.join(root, 'dataset_info.json'), 'r', encoding='utf-8') as f:
        info = json.load(f)
    with gzip.open(os.path.join(root, 'cryptobench_v1.jsonl.gz'), 'rb') as f:
        released = f.read()
    sizes = info['composition']
    fresh = release_payload(generate_cryptobench(seed=int(info['seed']), n_cipher=int(sizes['cipher']),
                                                 n_modular=int(sizes['modular']), n_hash=int(sizes['hash']))).encode('utf-8')
    checks = [('released file matches the recorded checksum', hashlib.sha256(released).hexdigest() == info['sha256_uncompressed']),
              ('fresh generation is byte-identical to the released file', fresh == released),
              ('instance count matches', released.count(b'\n') == int(info['n_instances']))]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in checks) else 1


def _for_each(cfg, args, fn):
    from experiments.model import load_model_and_tokenizer
    out = {}
    for m in _models(cfg, args):
        mt = load_model_and_tokenizer(cfg, cfg['models'][m])
        for b in _benchmarks(cfg, args):
            for cap in (args.caps or [generation_cap(cfg, b)]):
                out[f'{m}/{b}/T{cap}'] = fn(m, b, int(cap), mt)
    _print(out)
    return 0


def cmd_generate(args) -> int:
    from experiments.generation import run_generation
    cfg = _cfg(args)
    return _for_each(cfg, args, lambda m, b, cap, mt: {k: v for k, v in run_generation(
        cfg, m, b, args.split, cap=cap, model_and_tok=mt).items() if k not in ('dims', 'per_category')})


def cmd_measure_gap(args) -> int:
    from experiments.faithfulness import measure_gap, sample_size_sensitivity
    cfg = _cfg(args)
    if args.n_pairs is not None:
        sizes = [int(n) for n in args.n_pairs] or [int(n) for n in cfg.get('mine', {}).get('sample_sizes', [])]
        return _for_each(cfg, args, lambda m, b, cap, mt: sample_size_sensitivity(cfg, m, b, sizes, cap=cap, model_and_tok=mt))
    return _for_each(cfg, args, lambda m, b, cap, mt: {k: v for k, v in measure_gap(
        cfg, m, b, cap=cap, model_and_tok=mt).items() if k not in ('per_seed', 'dims', 'per_category')})


def cmd_perturbation(args) -> int:
    from experiments.perturbation import run_perturbation
    cfg = _cfg(args)
    return _for_each(cfg, args, lambda m, b, cap, mt: {k: run_perturbation(cfg, m, b, cap=cap, model_and_tok=mt)[k]
                                                       for k in ('PS_mean', 'PS_std')})


def cmd_patching(args) -> int:
    from experiments.patching import run_patching
    cfg = _cfg(args)
    return _for_each(cfg, args, lambda m, b, cap, mt: {k: run_patching(cfg, m, b, cap=cap, model_and_tok=mt)[k]
                                                       for k in ('R_bar', 'R_min_over_steps', 'R_max_over_steps', 'l_proj_bits', 'mechanism')})


def cmd_estimator_comparison(args) -> int:
    from experiments.faithfulness import estimator_comparison
    from experiments.model import load_model_and_tokenizer
    cfg = _cfg(args)
    model = _models(cfg, args)[0]
    benches = args.benchmarks or list(cfg.get('infonce', {}).get('benchmarks', []))
    mt = load_model_and_tokenizer(cfg, cfg['models'][model])
    _print(estimator_comparison(cfg, model, benches, model_and_tok=mt))
    return 0


def cmd_calibration(args) -> int:
    from estimation.calibration import run_calibration
    r = run_calibration(_cfg(args))
    _print({k: v for k, v in r.items() if k != 'settings'})
    return 0


def cmd_complexity(args) -> int:
    from data.cryptobench import complexity_report, generate_cryptobench
    from theory.complexity import gpqa_program_length, multiplication_running_example

    cfg = _cfg(args)
    sub = cfg['data']['benchmarks']['cryptobench']
    sizes = sub.get('sizes', {'cipher': 200, 'modular': 200, 'hash': 100})
    lz_method = str(cfg.get('complexity', {}).get('lz_method', 'lz78'))
    insts = generate_cryptobench(seed=int(sub.get('seed', 0)), n_cipher=int(sizes['cipher']), n_modular=int(sizes['modular']),
                                 n_hash=int(sizes['hash']))
    out = {'cryptobench': complexity_report(insts, lz_method=lz_method), 'running_example': multiplication_running_example(),
           'gpqa_program_length': gpqa_program_length()}
    try:
        from data.registry import get_benchmark
        from experiments.model import split_trace_and_answer
        from theory.complexity import gpqa_expert_trace_bits, gpqa_k_tilde_range
        traces = [split_trace_and_answer(i.trace)[0] for i in get_benchmark(cfg, 'gpqa_diamond').load('eval') if i.trace]
        expert = gpqa_expert_trace_bits(traces, method=lz_method, n_traces=int(cfg.get('complexity', {}).get('gpqa_expert_traces', 50)))
        out['gpqa_expert_lz'] = expert
        if expert['n']:
            out['gpqa_k_tilde_range'] = gpqa_k_tilde_range(out['gpqa_program_length'], expert)
    except (FileNotFoundError, KeyError):
        out['gpqa_expert_lz'] = {'unavailable': 'run prepare-data --benchmarks gpqa_diamond first'}
    d = os.path.join(cfg.get('paths', {}).get('results', 'results'), 'complexity')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'complexity.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    _print(out)
    return 0


def cmd_ib_frontier(args) -> int:
    from experiments.ib import run_frontier
    cfg = _cfg(args)
    _print(run_frontier(cfg, _models(cfg, args)[0], bench_names=args.benchmarks))
    return 0


def cmd_tables(args) -> int:
    from experiments import tables as T

    cfg = _cfg(args)
    models, benches = (_models(cfg, args), _benchmarks(cfg, args))
    primary = models[0]
    tabs = [T.main_results(cfg, primary, benches), T.perturbation_by_model(cfg, models, benches),
            T.complexity_ordering(cfg, models, benches)]
    caps = list(cfg.get('scaling', {}).get('caps', [128, 256, 512, 1024, 2048]))
    if 'cryptobench' in benches:
        tabs.append(T.scaling(cfg, primary, 'cryptobench', caps))
    if len(models) > 1:
        tabs.append(T.cross_model(cfg, models, [b for b in ('cryptobench', 'gpqa_diamond') if b in benches]))
    gap = T._read(T.gap_path(cfg, primary, 'cryptobench', generation_cap(cfg, 'cryptobench')))
    if gap is not None and 'dims' in gap:
        tabs.append(T.recovery_sensitivity(generation_cap(cfg, 'cryptobench'), int(gap['dims']['v']), int(gap['dims']['d']),
                                           gap['delta_mean'], list(cfg.get('sensitivity', {}).get('R_grid', [0.5, 0.8, 1.2, 1.6, 2.0]))))
    tabs.append(T.frontier(cfg, primary))
    tabs.append(T.estimator_agreement(cfg, primary))
    path = T.write_tables(cfg, tabs)
    with open(path, 'r', encoding='utf-8') as f:
        print(f.read())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python main.py')
    sub = p.add_subparsers(dest='command', required=True)

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument('--config', default='configs/base.yaml')
        s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
        s.set_defaults(func=fn)
        return s

    s = add('prepare-data', cmd_prepare_data, 'download and convert the public benchmarks')
    s.add_argument('--benchmarks', nargs='+', default=None)
    s.add_argument('--hf-token', default=None)
    add('make-cryptobench', cmd_make_cryptobench, 'generate the CryptoBench splits')
    s = add('verify-cryptobench', cmd_verify_cryptobench, 'check the released instances against a fresh generation')
    s.add_argument('--release-dir', default=None)
    for name, fn, h in (('generate', cmd_generate, 'chain-of-thought generation'),
                        ('measure-gap', cmd_measure_gap, 'critic scores and the gap'),
                        ('perturbation', cmd_perturbation, 'perturbation sensitivity'),
                        ('patching', cmd_patching, 'attention-recovery estimate')):
        s = add(name, fn, h)
        s.add_argument('--models', nargs='+', default=None)
        s.add_argument('--benchmarks', nargs='+', default=None)
        s.add_argument('--caps', nargs='+', default=None)
        if name == 'generate':
            s.add_argument('--split', default='eval', choices=['eval', 'train', 'validation'])
        if name == 'measure-gap':
            s.add_argument('--n-pairs', nargs='*', default=None)
    for name, fn, h in (('estimator-comparison', cmd_estimator_comparison, 'MINE and InfoNCE rank agreement'),
                        ('ib-frontier', cmd_ib_frontier, 'bottleneck fine-tuning sweep'),
                        ('tables', cmd_tables, 'aggregate results into tables')):
        s = add(name, fn, h)
        s.add_argument('--models', nargs='+', default=None)
        s.add_argument('--benchmarks', nargs='+', default=None)
    add('calibration', cmd_calibration, 'critic calibration on synthetic pairs')
    add('complexity', cmd_complexity, 'solution-trace complexity estimates')
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
