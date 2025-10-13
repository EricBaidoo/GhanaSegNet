#!/usr/bin/env python3
"""
Enhanced Multi-Model Evaluation Script for GhanaSegNet Research Project
Evaluates trained baseline models: UNet, DeepLabV3+, SegFormer-B0, GhanaSegNet (Enhanced)
Author: EricBaidoo
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path for module imports
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Import all baseline models
from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormerB0
from models.ghanasegnet import GhanaSegNet

# Import utilities
from data.dataset_loader import GhanaFoodDataset
from utils.metrics import compute_iou, compute_pixel_accuracy, compute_f1_per_class
from utils.losses import CombinedLoss

def get_model_and_criterion(model_name, num_classes=6):
    """
    Initialize model and its original loss function for evaluation.
    """
    models = {
        'unet': lambda: UNet(n_channels=3, n_classes=num_classes),
        'deeplabv3plus': lambda: DeepLabV3Plus(num_classes=num_classes),
        'segformer': lambda: SegFormerB0(num_classes=num_classes),
        'ghanasegnet': lambda: GhanaSegNet(num_classes=num_classes)
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(models.keys())}")
    
    model = models[model_name.lower()]()
    return model

def evaluate_model(model, test_loader, device, model_name, num_classes=6, save_preds=False, results_dir=None):
    """
    Comprehensive model evaluation with detailed metrics and optional prediction saving.
    """
    model.eval()
    total_iou, total_acc = 0.0, 0.0
    all_preds = []
    all_labels = []
    batch_losses = []
    
    # Class names for Ghana food dataset
    class_names = ['Background', 'Rice', 'Stew', 'Protein', 'Vegetables', 'Other']
    
    print(f"\n🔍 Evaluating {model_name.upper()} model...")
    print("=" * 60)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Evaluating [{model_name}]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Compute batch metrics
            batch_iou = compute_iou(preds, masks, num_classes)
            batch_acc = compute_pixel_accuracy(preds, masks)
            
            total_iou += batch_iou
            total_acc += batch_acc
            
            # Store predictions and labels for detailed analysis
            all_preds.append(preds.cpu())
            all_labels.append(masks.cpu())
            
            # Save predictions if requested
            if save_preds and results_dir:
                pred_dir = Path(results_dir) / "predictions" / model_name
                pred_dir.mkdir(parents=True, exist_ok=True)
                
                for i in range(preds.size(0)):
                    pred_path = pred_dir / f"batch_{batch_idx:04d}_sample_{i:02d}.npy"
                    np.save(pred_path, preds[i].cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'IoU': f'{batch_iou:.4f}',
                'Acc': f'{batch_acc:.4f}'
            })
    
    # Compute overall metrics
    avg_iou = total_iou / len(test_loader)
    avg_acc = total_acc / len(test_loader)
    
    # Concatenate all predictions and labels for detailed analysis
    preds_cat = torch.cat(all_preds).numpy().flatten()
    labels_cat = torch.cat(all_labels).numpy().flatten()
    
    # Compute per-class F1 scores
    f1_scores = compute_f1_per_class(torch.from_numpy(preds_cat), torch.from_numpy(labels_cat), num_classes)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels_cat, preds_cat, labels=list(range(num_classes)))
    
    # Generate classification report
    report = classification_report(
        labels_cat, 
        preds_cat, 
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Print detailed results
    print(f"\n📊 EVALUATION RESULTS - {model_name.upper()}")
    print("=" * 60)
    print(f"📈 Overall Metrics:")
    print(f"   Mean IoU:        {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    print(f"   Pixel Accuracy:  {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"   Total Samples:   {len(test_loader.dataset)}")
    
    print(f"\n🎯 Per-Class Performance:")
    for i, (class_name, f1_score) in enumerate(zip(class_names, f1_scores)):
        precision = report[str(i)]['precision']
        recall = report[str(i)]['recall']
        support = int(report[str(i)]['support'])
        print(f"   {class_name:<12}: F1={f1_score:.3f}, P={precision:.3f}, R={recall:.3f} (n={support:,})")
    
    # Prepare results dictionary
    results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(test_loader.dataset),
        'num_classes': num_classes,
        'overall_metrics': {
            'mean_iou': float(avg_iou),
            'pixel_accuracy': float(avg_acc),
            'mean_f1': float(np.mean(f1_scores))
        },
        'per_class_metrics': {
            class_names[i]: {
                'f1_score': float(f1_scores[i]),
                'precision': float(report[str(i)]['precision']),
                'recall': float(report[str(i)]['recall']),
                'support': int(report[str(i)]['support'])
            } for i in range(num_classes)
        },
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    return results


def save_results_to_json(results, output_path):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {output_path}")

def generate_evaluation_report(results, output_path):
    """Generate a human-readable evaluation report."""
    with open(output_path, 'w') as f:
        f.write(f"# GhanaSegNet Evaluation Report\n\n")
        f.write(f"**Model:** {results['model_name'].upper()}\n")
        f.write(f"**Evaluation Date:** {results['timestamp']}\n")
        f.write(f"**Dataset Size:** {results['dataset_size']:,} samples\n\n")
        
        f.write(f"## Overall Performance\n\n")
        f.write(f"| Metric | Score | Percentage |\n")
        f.write(f"|--------|-------|------------|\n")
        f.write(f"| Mean IoU | {results['overall_metrics']['mean_iou']:.4f} | {results['overall_metrics']['mean_iou']*100:.2f}% |\n")
        f.write(f"| Pixel Accuracy | {results['overall_metrics']['pixel_accuracy']:.4f} | {results['overall_metrics']['pixel_accuracy']*100:.2f}% |\n")
        f.write(f"| Mean F1 Score | {results['overall_metrics']['mean_f1']:.4f} | {results['overall_metrics']['mean_f1']*100:.2f}% |\n\n")
        
        f.write(f"## Per-Class Performance\n\n")
        f.write(f"| Class | F1 Score | Precision | Recall | Support |\n")
        f.write(f"|-------|----------|-----------|--------|---------|\n")
        
        for class_name, metrics in results['per_class_metrics'].items():
            f.write(f"| {class_name} | {metrics['f1_score']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['support']:,} |\n")
        
        f.write(f"\n## Model Architecture\n\n")
        if 'ghanasegnet' in results['model_name']:
            f.write("This model uses a novel CNN-Transformer hybrid architecture specifically designed for Ghanaian food segmentation.\n")
        else:
            f.write(f"This is a baseline {results['model_name'].upper()} model for comparison.\n")
    
    print(f"📋 Report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models for GhanaSegNet research')
    parser.add_argument('--model', type=str, required=True,
                       choices=['unet', 'deeplabv3plus', 'segformer', 'ghanasegnet'],
                       help='Model architecture to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation (default: 4)')
    parser.add_argument('--num-classes', type=int, default=6,
                       help='Number of classes (default: 6)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save prediction masks as .npy files')
    parser.add_argument('--results-dir', type=str, default='evaluation_results',
                       help='Directory to save results (default: evaluation_results)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loader workers (default: 2)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting evaluation on {device}")
    print(f"📝 Model: {args.model.upper()}")
    print(f"💾 Checkpoint: {args.checkpoint}")
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        model = get_model_and_criterion(args.model, args.num_classes)
        model = model.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded checkpoint weights")
            
    except Exception as e:
        print(f"❌ Error loading model or checkpoint: {e}")
        return
    
    # Load test dataset
    try:
        test_dataset = GhanaFoodDataset(split='test')
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        print(f"📚 Test dataset: {len(test_dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Run evaluation
    try:
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            model_name=args.model,
            num_classes=args.num_classes,
            save_preds=args.save_predictions,
            results_dir=results_dir
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_path = results_dir / f"{args.model}_evaluation_{timestamp}.json"
        save_results_to_json(results, json_path)
        
        # Markdown report
        report_path = results_dir / f"{args.model}_report_{timestamp}.md"
        generate_evaluation_report(results, report_path)
        
        # Summary
        print(f"\n🎉 EVALUATION COMPLETE!")
        print(f"📊 Final Scores:")
        print(f"   • Mean IoU: {results['overall_metrics']['mean_iou']:.4f} ({results['overall_metrics']['mean_iou']*100:.2f}%)")
        print(f"   • Pixel Accuracy: {results['overall_metrics']['pixel_accuracy']:.4f} ({results['overall_metrics']['pixel_accuracy']*100:.2f}%)")
        print(f"   • Mean F1: {results['overall_metrics']['mean_f1']:.4f} ({results['overall_metrics']['mean_f1']*100:.2f}%)")
        print(f"📁 Results saved in: {results_dir}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
