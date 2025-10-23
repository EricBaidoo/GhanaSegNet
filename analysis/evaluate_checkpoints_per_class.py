"""
Evaluate saved checkpoints to produce per-class IoU using the repository's `scripts/evaluate.py`.

Usage (PowerShell):
  python analysis\evaluate_checkpoints_per_class.py --checkpoints checkpoints --out results/per_class_summary.json

Note: this script shells out to `python scripts/evaluate.py --model <model> --checkpoint <ckpt> --out results/<model>_eval.json`
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path


def evaluate_all(checkpoint_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # find directories directly under checkpoint_root
    for model_dir in glob.glob(os.path.join(checkpoint_root, '*')):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)
        ckpt = os.path.join(model_dir, 'best_model.pth')
        if not os.path.isfile(ckpt):
            print(f'No best_model.pth for {model_name} in {model_dir}, skipping')
            continue

        # Create a temporary results dir for this evaluation
        temp_results = Path('results') / 'tmp_eval' / model_name
        temp_results.mkdir(parents=True, exist_ok=True)

        # Call evaluate.py with the --results-dir argument (evaluate.py will create timestamped JSONs)
        cmd = [
            'python', 'scripts/evaluate.py',
            '--model', model_name,
            '--checkpoint', ckpt,
            '--results-dir', str(temp_results)
        ]
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)

        # Find the generated evaluation JSON in the temp_results directory and move/rename it to out_dir
        generated = list(temp_results.glob(f"{model_name}_evaluation_*.json"))
        if not generated:
            print(f'Warning: no evaluation JSON produced for {model_name} in {temp_results}')
            continue
        # If multiple, pick the latest by modification time
        generated.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        src = generated[0]
        dest = Path(out_dir) / f"{model_name}_eval.json"
        shutil.move(str(src), str(dest))
        print(f'Moved {src} -> {dest}')
        # clean up temp folder for this model
        try:
            shutil.rmtree(temp_results)
        except Exception:
            pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', default='checkpoints')
    p.add_argument('--out', default='results/per_class_summary')
    args = p.parse_args()
    evaluate_all(args.checkpoints, args.out)
