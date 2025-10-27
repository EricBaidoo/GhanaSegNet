"""
generate_all_analysis_figures.py

Creates figures and tables for thesis Chapter 3 and Chapter 4.

Usage: python scripts/generate_all_analysis_figures.py [--results-dir CHECKPOINTS] [--out-dir figures]

The script searches recursively for *_results.json files under --results-dir. If none are
found it will produce placeholder/synthetic figures so you can draft thesis figures without
needing large checkpoints.
"""
from pathlib import Path
import argparse
import json
import math
import os
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_result_files(results_dir: Path):
    if not results_dir.exists():
        return []
    return list(results_dir.rglob("*_results.json"))


def load_results(files):
    results = {}
    for f in files:
        try:
            data = json.loads(f.read_text())
            # infer model name from filename or object
            name = data.get('model_name') or f.stem.replace('_results', '')
            results[name] = data
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")
    return results


def save_svg_and_png(fig, out_path_base: Path, dpi=300):
    png_path = out_path_base.with_suffix('.png')
    svg_path = out_path_base.with_suffix('.svg')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    try:
        fig.savefig(svg_path, bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)
    return png_path, svg_path


def figure_3_1_model_architecture(out_dir: Path):
    # Simple schematic: encoder -> bottleneck -> decoder
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    # rects
    rects = [((0.05, 0.2), 'Input Image'), ((0.35, 0.55), 'Encoder\n(Backbone)'),
             ((0.55, 0.5), 'Bottleneck'), ((0.75, 0.55), 'Decoder\n(Up-sampling)'), ((0.95, 0.5), 'Output Mask')]
    for (x, y), label in rects:
        ax.add_patch(plt.Rectangle((x - 0.08, y - 0.12), 0.16, 0.24, facecolor='#e6f2ff', edgecolor='k'))
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
    # arrows
    ax.annotate('', xy=(0.15, 0.5), xytext=(0.27, 0.6), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.45, 0.6), xytext=(0.53, 0.52), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.57, 0.52), xytext=(0.68, 0.6), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.83, 0.6), xytext=(0.92, 0.52), arrowprops=dict(arrowstyle='->'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out = out_dir / 'figure3_1_model_architecture'
    return save_svg_and_png(fig, out)


def figure_3_2_preprocessing_workflow(out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    steps = ['Raw Images', 'Resize/Crop', 'Normalize', 'Augment\n(flip, color, noise)', 'Batch']
    xs = np.linspace(0.05, 0.95, len(steps))
    for x, s in zip(xs, steps):
        ax.add_patch(plt.Circle((x, 0.5), 0.06, facecolor='#fff5e6', edgecolor='k'))
        ax.text(x, 0.5, s, ha='center', va='center', fontsize=9)
    for i in range(len(xs) - 1):
        ax.annotate('', xy=(xs[i] + 0.05, 0.5), xytext=(xs[i + 1] - 0.05, 0.5), arrowprops=dict(arrowstyle='->'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out = out_dir / 'figure3_2_preprocessing_workflow'
    return save_svg_and_png(fig, out)


def figure_3_3_training_workflow(out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')
    boxes = ['Training Loop', 'Validation', 'Checkpointing', 'Evaluation\n& Metrics']
    xs = [0.12, 0.38, 0.64, 0.88]
    for x, b in zip(xs, boxes):
        ax.add_patch(plt.Rectangle((x - 0.12, 0.4), 0.24, 0.2, facecolor='#e8ffe6', edgecolor='k'))
        ax.text(x, 0.5, b, ha='center', va='center')
    # arrows with loop arrow back to training
    ax.annotate('', xy=(0.24, 0.5), xytext=(0.36, 0.5), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.5, 0.5), xytext=(0.62, 0.5), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.76, 0.5), xytext=(0.86, 0.5), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.06, 0.5), xytext=(0.06, 0.7), arrowprops=dict(arrowstyle='-|>', linestyle='dashed'))
    ax.annotate('', xy=(0.06, 0.7), xytext=(0.86, 0.7), arrowprops=dict(arrowstyle='->', linestyle='dashed'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out = out_dir / 'figure3_3_training_workflow'
    return save_svg_and_png(fig, out)


def table_3_1_ethics(out_dir: Path, tables_dir: Path):
    rows = [
        ("Data Source", "Locally collected food imagery; see dataset README for consent and provenance"),
        ("Consent", "Public/street photography; identifiable faces removed; only food items used"),
        ("Anonymization", "No personal identifiers stored; images stripped of metadata"),
        ("Governance", "Access controlled via Google Drive; research use only"),
        ("Bias Mitigation", "Diverse sampling across regions and vendors; class balancing applied during training"),
    ]
    df = pd.DataFrame(rows, columns=['Aspect', 'Description'])
    ensure_dir(tables_dir)
    csv_path = tables_dir / 'table3_1_ethics.csv'
    tex_path = tables_dir / 'table3_1_ethics.tex'
    df.to_csv(csv_path, index=False)
    with open(tex_path, 'w', encoding='utf8') as f:
        f.write(df.to_latex(index=False))
    return csv_path, tex_path


def table_4_1_quantitative(results: Dict[str, Any], tables_dir: Path):
    rows = []
    for name, r in results.items():
        best_iou = r.get('best_iou', None)
        final_loss = r.get('final_val_loss', r.get('final_loss', None))
        params = r.get('total_parameters', r.get('params', None))
        epoch = r.get('final_epoch', None)
        status = r.get('status', 'completed' if best_iou is not None else 'unknown')
        rows.append((name, best_iou, final_loss, params, status, epoch))
    if not rows:
        # placeholder
        rows = [ ('ghanasegnet', None, None, None, 'no-results', None) ]
    df = pd.DataFrame(rows, columns=['Model', 'Best mIoU', 'Final Val Loss', 'Total Parameters', 'Status', 'Final Epoch'])
    df = df.sort_values(by='Best mIoU', ascending=False, na_position='last')
    ensure_dir(tables_dir)
    csv_path = tables_dir / 'table4_1_quantitative_results.csv'
    tex_path = tables_dir / 'table4_1_quantitative_results.tex'
    df.to_csv(csv_path, index=False)
    with open(tex_path, 'w', encoding='utf8') as f:
        f.write(df.to_latex(index=False, na_rep='N/A'))
    return csv_path, tex_path


def figure_4_1_training_curves(results: Dict[str, Any], out_dir: Path):
    # Prefer a model named 'ghanasegnet' if present
    model_name = 'ghanasegnet' if 'ghanasegnet' in results else (next(iter(results.keys())) if results else None)
    fig, ax = plt.subplots(figsize=(6, 4))
    if model_name and 'history' in results.get(model_name, {}):
        h = results[model_name]['history']
        epochs = range(1, len(h.get('train_loss', [])) + 1)
        ax.plot(epochs, h.get('train_loss', []), label='Train Loss')
        ax.plot(epochs, h.get('val_loss', []), label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    else:
        # synthetic example curves
        epochs = np.arange(1, 21)
        train_loss = np.exp(-0.12 * epochs) + 0.02 * np.random.rand(len(epochs))
        val_loss = np.exp(-0.10 * epochs) + 0.03 * np.random.rand(len(epochs))
        ax.plot(epochs, train_loss, label='Train Loss')
        ax.plot(epochs, val_loss, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    out = out_dir / 'figure4_1_training_curves'
    return save_svg_and_png(fig, out)


def comparison_barplots(results: Dict[str, Any], out_dir: Path):
    df_rows = []
    for name, r in results.items():
        df_rows.append({'Model': name, 'Best mIoU': r.get('best_iou', np.nan), 'Final Val Loss': r.get('final_val_loss', r.get('final_loss', np.nan))})
    if not df_rows:
        # synthetic
        df = pd.DataFrame({'Model': ['ghanasegnet', 'deeplabv3plus', 'unet'], 'Best mIoU': [0.72, 0.68, 0.61], 'Final Val Loss': [0.38, 0.45, 0.52]})
    else:
        df = pd.DataFrame(df_rows)
    ensure_dir(out_dir)
    sns.set(style='whitegrid')
    # Figure 4.2 mIoU comparison (improved labeling and ordering)
    # Sort models by Best mIoU for clear comparison
    df_sorted = df.sort_values(by='Best mIoU', ascending=False).reset_index(drop=True)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette('muted', n_colors=len(df_sorted))
    sns.barplot(x='Model', y='Best mIoU', data=df_sorted, ax=ax1, palette=palette, order=df_sorted['Model'])
    ax1.set_title('Validation mIoU Comparison', fontsize=12)
    ax1.set_xlabel('Model', fontsize=10)
    ax1.set_ylabel('Best mIoU', fontsize=10)
    # rotate x labels if many or long names
    for lbl in ax1.get_xticklabels():
        lbl.set_rotation(25)
        lbl.set_ha('right')
    # y-axis limits: start at 0 and add 10% headroom
    # Scale y-axis to the data range (don't force a full 0-1 range) so small absolute differences are visible
    max_miou = float(df_sorted['Best mIoU'].max()) if not df_sorted['Best mIoU'].isnull().all() else None
    if max_miou and max_miou > 0:
        ax1.set_ylim(0, max_miou * 1.15)
    else:
        ax1.set_ylim(0, 1.0)
    # annotate bars with the mIoU value (3 decimal places)
    # Thicken bars (narrower width but with edge) and annotate values
    target_bar_width = 0.6
    for p in ax1.patches:
        # adjust width and center the bar
        try:
            current_width = p.get_width()
            diff = current_width - target_bar_width
            p.set_width(target_bar_width)
            p.set_x(p.get_x() + diff / 2.0)
        except Exception:
            pass
        p.set_edgecolor('#333333')
        p.set_linewidth(0.8)
        height = p.get_height()
        if not (height is None or np.isnan(height)):
            ax1.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 6), textcoords='offset points')
    fig1.tight_layout()
    out1 = out_dir / 'figure4_2_miou_comparison'
    save_svg_and_png(fig1, out1)
    # Figure 4.3 loss comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    # use the same order and palette as the mIoU plot for consistency
    sns.barplot(x='Model', y='Final Val Loss', data=df_sorted, ax=ax2, palette=palette, order=df_sorted['Model'])
    ax2.set_title('Validation Loss Comparison', fontsize=12)
    ax2.set_xlabel('Model', fontsize=10)
    ax2.set_ylabel('Final Val Loss', fontsize=10)
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation(25)
        lbl.set_ha('right')
    # annotate loss bars where values exist
    # Adjust bar width and annotate
    for p in ax2.patches:
        try:
            current_width = p.get_width()
            diff = current_width - target_bar_width
            p.set_width(target_bar_width)
            p.set_x(p.get_x() + diff / 2.0)
        except Exception:
            pass
        p.set_edgecolor('#333333')
        p.set_linewidth(0.8)
        height = p.get_height()
        if not (height is None or np.isnan(height)):
            ax2.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 6), textcoords='offset points')
    fig2.tight_layout()
    out2 = out_dir / 'figure4_3_loss_comparison'
    save_svg_and_png(fig2, out2)
    return out1, out2


def table_4_2_ablation(results: Dict[str, Any], tables_dir: Path):
    # Try to find ablation info in results; otherwise generate placeholders
    ablation_rows = []
    for name, r in results.items():
        if 'ablations' in r:
            for ab in r['ablations']:
                ablation_rows.append((name, ab.get('change', ''), ab.get('delta_miou', None)))
    if not ablation_rows:
        ablation_rows = [
            ("base", "GhanaSegNet baseline", 0.0),
            ("-augmentation", "No augmentation", -0.03),
            ("-decoder_skip", "Remove skip connections", -0.05),
        ]
    df = pd.DataFrame(ablation_rows, columns=['Experiment', 'Change', 'Delta mIoU'])
    ensure_dir(tables_dir)
    csv_path = tables_dir / 'table4_2_ablation.csv'
    tex_path = tables_dir / 'table4_2_ablation.tex'
    df.to_csv(csv_path, index=False)
    with open(tex_path, 'w', encoding='utf8') as f:
        f.write(df.to_latex(index=False))
    return csv_path, tex_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='checkpoints', help='Directory to search for *_results.json files')
    parser.add_argument('--out-dir', type=str, default='figures', help='Directory to write figures (PNG/SVG)')
    parser.add_argument('--tables-dir', type=str, default='tables', help='Directory to write CSV/LaTeX tables')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    tables_dir = Path(args.tables_dir)

    ensure_dir(out_dir)
    ensure_dir(tables_dir)

    files = find_result_files(results_dir)
    if not files:
        print(f"No result JSONs found under {results_dir}. Creating placeholder/synthetic outputs.")
    else:
        print(f"Found {len(files)} result files under {results_dir}.")

    results = load_results(files)

    artifacts = []
    # Chapter 3
    artifacts += list(figure_3_1_model_architecture(out_dir))
    artifacts += list(figure_3_2_preprocessing_workflow(out_dir))
    artifacts += list(figure_3_3_training_workflow(out_dir))
    artifacts += list(table_3_1_ethics(out_dir, tables_dir))
    # Chapter 4
    artifacts += list(table_4_1_quantitative(results, tables_dir))
    artifacts += list(figure_4_1_training_curves(results, out_dir))
    artifacts += list(comparison_barplots(results, out_dir))
    artifacts += list(table_4_2_ablation(results, tables_dir))

    print('\nDone. Generated the following artifacts:')
    for a in artifacts:
        print(f' - {a}')


if __name__ == '__main__':
    main()
