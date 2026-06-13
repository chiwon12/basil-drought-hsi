# -*- coding: utf-8 -*-
"""
Full plant-aware sweep orchestrator.

Matrix per sensor: 3 norm × 3 data × 5 model = 45 combos
  norm  ∈ {none, snv, msc}
  data  ∈ {raw, ae, gan}
  model ∈ {knn, svm, rf, xgb, cnn}

Each combo runs as a resumable subprocess of run_single_experiment.py (skips if
its test_metrics.json already exists). ML and CNN both use Optuna TPE with 200
trials, no pruner, selection = mean weighted-F1 (paper Table S1). `--mode` lets
you split the run across processes/GPUs (e.g. cnn on GPU, ml on CPU).

Usage:
  python run_all_experiments.py --sensor sw
  python run_all_experiments.py --sensor vn --mode cnn --gpu 0
  python run_all_experiments.py --sensor sw --mode mlnoxgb   # cpu, no xgb
"""
import argparse
import itertools
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]               # repo root; src/training/ -> parents[1]
SCRIPT = HERE / 'run_single_experiment.py'

NORMS = ['none', 'snv', 'msc']
DATAS = ['raw', 'ae', 'gan']
MODELS = ['knn', 'svm', 'rf', 'xgb', 'cnn']
MODE_MODELS = {'all': MODELS, 'cnn': ['cnn'], 'ml': ['knn', 'svm', 'rf', 'xgb'],
               'mlnoxgb': ['knn', 'svm', 'rf'], 'xgb': ['xgb']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensor', choices=['sw', 'vn'], default='sw')
    ap.add_argument('--mode', choices=list(MODE_MODELS), default='all')
    ap.add_argument('--datas', default='raw,ae,gan', help='comma list to restrict data axis')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--out_root', default=None, help='results root (default: repo/results)')
    args = ap.parse_args()

    out_root = Path(args.out_root) if args.out_root else REPO / 'results'
    rdir = out_root / 'per_combo'
    sel = MODE_MODELS[args.mode]
    datas = args.datas.split(',')
    # CNN first (GPU), then ML (single-process ordering; --mode can isolate one).
    order = [(n, d, 'cnn') for n, d in itertools.product(NORMS, datas) if 'cnn' in sel] + \
            [(n, d, m) for n, d, m in itertools.product(NORMS, datas, MODELS)
             if m != 'cnn' and m in sel]

    print(f'[orch] sensor={args.sensor} {len(order)} combos -> {rdir}', flush=True)
    t0 = time.time()
    for i, (n, d, m) in enumerate(order, 1):
        combo = f'{args.sensor}_{n}_{d}_{m}'
        if (rdir / combo / 'test_metrics.json').exists():
            print(f'[{i}/{len(order)}] skip {combo}', flush=True); continue
        cmd = [os.environ.get('PYTHON', 'python3'), str(SCRIPT),
               '--sensor', args.sensor, '--norm', n, '--data', d, '--model', m,
               '--seed', str(args.seed), '--out_root', str(out_root)]
        if args.gpu is not None:
            cmd += ['--gpu', str(args.gpu)]
        t = time.time()
        rc = subprocess.run(cmd).returncode
        print(f'[{i}/{len(order)}] {combo} rc={rc} ({time.time() - t:.0f}s, '
              f'total {(time.time() - t0) / 60:.1f}m)', flush=True)
    print(f'[orch] done {(time.time() - t0) / 60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
