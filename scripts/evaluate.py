import sys
import os
# Add project root to Python path for module imports
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from models.unet import UNet  # Using UNet for evaluation
from data.dataset_loader import GhanaFoodDataset
from utils.metrics import compute_iou, compute_pixel_accuracy, compute_f1_per_class

def evaluate(model, test_loader, device, num_classes=6, save_preds=False):
    model.eval()
    total_iou, total_acc = 0.0, 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            total_iou += compute_iou(preds, masks, num_classes)
            total_acc += compute_pixel_accuracy(preds, masks)

            all_preds.append(preds.cpu())
            all_labels.append(masks.cpu())

            if save_preds:
                for i in range(preds.size(0)):
                    np.save(f"results/pred_{i}.npy", preds[i].numpy())

    avg_iou = total_iou / len(test_loader)
    avg_acc = total_acc / len(test_loader)

    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    f1_scores = compute_f1_per_class(preds_cat, labels_cat, num_classes)

    print("\n Evaluation Results")
    print(f"Mean IoU: {avg_iou:.4f}")
    print(f"Pixel Accuracy: {avg_acc:.4f}")
    print(f"Per-class F1 Scores: {np.round(f1_scores, 3)}")

    return avg_iou, avg_acc, f1_scores


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(num_classes=6).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))

    test_dataset = GhanaFoodDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    os.makedirs("results", exist_ok=True)
    evaluate(model, test_loader, device, num_classes=6, save_preds=True)
