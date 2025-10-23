import json
import glob
import os
import math
from statistics import mean, pstdev

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
files = glob.glob(os.path.join(RESULTS_DIR, '*_results.json'))

out = []
for f in files:
    with open(f, 'r') as fh:
        data = json.load(fh)
    model = data.get('model_name', os.path.basename(f))
    history = data.get('training_history', [])
    val_iou_vals = [h.get('val_iou') for h in history if 'val_iou' in h]
    n = len(val_iou_vals)
    if n == 0:
        continue
    mu = mean(val_iou_vals)
    # population std dev (pstdev) to be conservative for small samples
    sd = pstdev(val_iou_vals)
    se = sd / math.sqrt(n)
    # try to use t critical if available
    try:
        from scipy import stats
        tcrit = stats.t.ppf(0.975, df=n-1)
    except Exception:
        tcrit = 1.96
    ci_low = mu - tcrit * se
    ci_high = mu + tcrit * se
    out.append({
        'model': model,
        'n_epochs': n,
        'mean_val_iou': mu,
        'std_val_iou': sd,
        'ci95_low': ci_low,
        'ci95_high': ci_high,
        'best_val_iou': data.get('best_iou')
    })

# Print a markdown table and a short JSON for programmatic use
print('| Model | n_epochs | mean val IoU | std | 95% CI | best recorded IoU |')
print('|------:|---------:|-------------:|----:|:------:|:-----------------:|')
for r in out:
    print(f"| {r['model']} | {r['n_epochs']} | {r['mean_val_iou']:.6f} ({r['mean_val_iou']*100:.2f}%) | {r['std_val_iou']:.6f} | ({r['ci95_low']:.6f}, {r['ci95_high']:.6f}) | {r['best_val_iou']:.6f} |")

# Save JSON summary
with open(os.path.join(RESULTS_DIR, 'val_iou_stats_summary.json'), 'w') as fh:
    json.dump(out, fh, indent=2)

print('\nSaved summary to results/val_iou_stats_summary.json')
