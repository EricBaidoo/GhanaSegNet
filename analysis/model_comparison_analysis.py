import json
import os
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FILES = [
    'deeplabv3plus_results.json',
    'ghanasegnet_results.json',
    'segformer_results.json',
    'unet_results.json'
]

models = []
for fname in FILES:
    with open(os.path.join(RESULTS_DIR, fname), 'r') as f:
        models.append(json.load(f))

# Plotting
metrics = ['val_iou', 'val_loss', 'val_accuracy']
for metric in metrics:
    plt.figure(figsize=(8, 5))
    for m in models:
        epochs = [h['epoch'] for h in m['training_history']]
        values = [h[metric] for h in m['training_history']]
        plt.plot(epochs, values, label=m['model_name'])
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{metric}_comparison.png'))
    plt.close()

# Print summary
print('Model Comparison Summary:')
for m in models:
    print(f"Model: {m['model_name']}")
    print(f"  Best IoU: {m['best_iou']:.4f}")
    print(f"  Final Epoch: {m['final_epoch']}")
    print(f"  Parameters: {m['trainable_parameters']:,}")
    print()
print('Plots saved in results folder.')
