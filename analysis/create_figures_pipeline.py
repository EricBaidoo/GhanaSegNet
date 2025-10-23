import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import os

# Figure 3.1: Pipeline Overview (Flowchart)
def create_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 2))
    stages = ['Data', 'Preprocessing', 'Model Init', 'Training', 'Evaluation', 'Artifacts']
    for i, stage in enumerate(stages):
        ax.add_patch(mpatches.FancyBboxPatch((i*1.5, 0.5), 1.3, 0.7, boxstyle="round,pad=0.1", fc="#e0e0e0"))
        ax.text(i*1.5+0.65, 0.85, stage, ha='center', va='center', fontsize=12)
        if i < len(stages)-1:
            ax.arrow(i*1.5+1.3, 0.85, 0.2, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('figures/figure3_1_pipeline.png')
    plt.close()

# Figure 3.2: Dataset Samples (Image + Mask)
def create_dataset_samples(image_path, mask_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(mpimg.imread(image_path))
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    axes[1].imshow(mpimg.imread(mask_path))
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('figures/figure3_2_dataset_samples.png')
    plt.close()

# Figure 3.3: Architecture Schematic (Block Diagram)
def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 1), 2, 1, boxstyle="round,pad=0.1", fc="#b3cde3"))
    ax.text(1.5, 1.5, 'Encoder (EfficientNet)', ha='center', va='center', fontsize=12)
    ax.add_patch(mpatches.FancyBboxPatch((3, 1), 2, 1, boxstyle="round,pad=0.1", fc="#fbb4ae"))
    ax.text(4, 1.5, 'Transformer Bottleneck', ha='center', va='center', fontsize=12)
    ax.add_patch(mpatches.FancyBboxPatch((5.5, 1), 2, 1, boxstyle="round,pad=0.1", fc="#ccebc5"))
    ax.text(6.5, 1.5, 'Decoder', ha='center', va='center', fontsize=12)
    # Arrows
    ax.arrow(2.5, 1.5, 0.5, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax.arrow(5, 1.5, 0.5, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('figures/figure3_3_architecture.png')
    plt.close()

# Figure 3.4: Training/Validation Curves
def create_training_curves(train_epochs, train_iou, val_iou):
    plt.figure(figsize=(7, 4))
    plt.plot(train_epochs, train_iou, label='Train IoU')
    plt.plot(train_epochs, val_iou, label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training/Validation IoU Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/figure3_4_training_curves.png')
    plt.close()

# Figure 3.4b: Qualitative Image Grid
def create_qualitative_grid(image_paths, mask_paths, pred_paths):
    n = min(3, len(image_paths))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3*n))
    for i in range(n):
        axes[i, 0].imshow(mpimg.imread(image_paths[i]))
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mpimg.imread(mask_paths[i]))
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(mpimg.imread(pred_paths[i]))
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig('figures/figure3_4_qualitative_grid.png')
    plt.close()

if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    create_pipeline_diagram()
    # For dataset samples, provide actual image/mask paths below:
    # create_dataset_samples('path/to/sample_image.jpg', 'path/to/sample_mask.png')
    create_architecture_diagram()
    # For training curves, provide actual epoch/IoU lists below:
    # create_training_curves([1,2,3,4,5], [0.1,0.15,0.2,0.22,0.24], [0.09,0.14,0.19,0.21,0.23])
    # For qualitative grid, provide lists of image/mask/prediction paths below:
    # create_qualitative_grid(['img1.jpg','img2.jpg','img3.jpg'], ['mask1.png','mask2.png','mask3.png'], ['pred1.png','pred2.png','pred3.png'])
