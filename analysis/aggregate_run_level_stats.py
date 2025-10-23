"""
Aggregate best-checkpoint metrics across multiple run JSONs per model.

Usage:
  python analysis/aggregate_run_level_stats.py --results-dir results --out results/aggregate_run_stats.json

This script looks for files named <model>_run*.json or <model>_results_*.json and computes mean/std/95% CI across the best_iou values per model.
"""
import argparse
import glob
import json
import math
import os
from statistics import mean, stdev

try:
    from scipy import stats
    has_scipy = True
except Exception:
    has_scipy = False


def tcrit95(n):
    if n <= 1 or not has_scipy:
        return 1.96
    return stats.t.ppf(0.975, df=n-1)


def aggregate(results_dir, out_path):
    files = glob.glob(os.path.join(results_dir, '*_results.json')) + glob.glob(os.path.join(results_dir, '*_run*.json'))
    runs_by_model = {}
    for f in files:
        with open(f, 'r') as fh:
            try:
                data = json.load(fh)
            except Exception:
                continue
        model = data.get('model_name') or os.path.basename(f).split('_')[0]
        runs_by_model.setdefault(model, []).append(data)

    aggregate = []
    for model, runs in runs_by_model.items():
        best_ious = [r.get('best_iou') for r in runs if 'best_iou' in r]
        n = len(best_ious)
        if n == 0:
            continue
        mu = mean(best_ious)
        sd = stdev(best_ious) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        t = tcrit95(n)
        ci_low = mu - t * se
        ci_high = mu + t * se
        aggregate.append({
            'model': model,
            'n_runs': n,
            'mean_best_iou': mu,
            'std_best_iou': sd,
            'ci95_low': ci_low,
            'ci95_high': ci_high,
            'runs': [os.path.basename(r.get('timestamp','')) for r in runs]
        })

    with open(out_path, 'w') as fh:
        json.dump(aggregate, fh, indent=2)
    print(f'Wrote aggregated stats to {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='results')
    p.add_argument('--out', default='results/aggregate_run_stats.json')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    aggregate(args.results_dir, args.out)
