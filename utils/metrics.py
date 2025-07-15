# Metric calculations (wrapped)
import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_iou(preds, labels, num_classes):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        label_inds = labels == cls
        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # Skip this class
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


def compute_pixel_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total


def compute_f1_per_class(preds, labels, num_classes):
    preds = preds.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    return f1_score(labels, preds, average=None, labels=list(range(num_classes)))
