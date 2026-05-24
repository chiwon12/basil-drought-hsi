# -*- coding: utf-8 -*-
"""
Part A orchestrator (CNN-only, 4 input modes).

Matrix (CNN only):
  - mode=all_leaf : data ∈ {raw, ae, gan}   → 3 × 3 norm × 2 sensor = 18
  - mode=c0/c1/c2 : data = raw only         → 3 × 3 norm × 2 sensor = 18
  Total: 36 CNN combos. Each on 2 GPUs (1 per GPU at a time).

ML 4종(KNN/SVM/RF/XGB)은 별도 orchestrator로 추후 실행.
"""
import itertools
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPT = HERE / 'run_single_experiment.py'
OUT = REPO / 'results' / 'all_experiments'
LOG_DIR = OUT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Override with environment variable if your interpreter lives elsewhere.
PYTHON = os.environ.get('PYTHON', 'python3')

SENSORS = ['sw', 'vn']
NORMS = ['none', 'snv', 'msc']
GPU_PARALLEL = 2  # one CNN per GPU at a time


def run_one(sensor, mode, norm, data, gpu):
    combo = f'{sensor}_{mode}_{norm}_{data}_cnn'
    log = LOG_DIR / f'{combo}.log'
    cmd = [PYTHON, str(SCRIPT),
           '--sensor', sensor, '--mode', mode, '--norm', norm, '--data', data,
           '--model', 'cnn', '--gpu', '0', '--out_root', str(OUT)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return combo, rc, time.time() - t0


def main():
    combos = []
    # all_leaf: raw / ae / gan
    for s, n, d in itertools.product(SENSORS, NORMS, ['raw', 'ae', 'gan']):
        combos.append((s, 'all_leaf', n, d))
    # c0 / c1 / c2: raw only
    for s, n, m in itertools.product(SENSORS, NORMS, ['c0', 'c1', 'c2']):
        combos.append((s, m, n, 'raw'))

    # Assign GPUs round-robin
    jobs = [(s, m, n, d, i % 2) for i, (s, m, n, d) in enumerate(combos)]
    print(f'[orchestrator] CNN jobs: {len(jobs)} (2 GPUs, {GPU_PARALLEL} parallel)')
    print(f'[orchestrator] out: {OUT}\n', flush=True)

    pool = ProcessPoolExecutor(max_workers=GPU_PARALLEL)
    futures = [pool.submit(run_one, *args) for args in jobs]
    total = len(futures)
    for done, fut in enumerate(as_completed(futures), 1):
        try:
            combo, rc, dur = fut.result()
            mark = 'OK' if rc == 0 else f'FAIL rc={rc}'
            print(f'[{done}/{total}] {combo}  {mark}  ({dur:.0f}s)', flush=True)
        except Exception as e:
            print(f'[{done}/{total}] error: {e}', flush=True)
    pool.shutdown()
    print(f'\n[done] master CSV: {OUT}/master.csv')


if __name__ == '__main__':
    main()
