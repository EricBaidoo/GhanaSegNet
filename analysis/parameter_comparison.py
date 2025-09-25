"""
Parameter Count Analysis for Fair Model Comparison
Analyzes parameter counts across all baseline models to ensure fair comparison

Author: EricBaidoo
"""

import torch
import torch.nn as nn
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Import all models
from unet import UNet
from deeplabv3plus import DeepLabV3Plus
from segformer import SegFormerB0
from ghanasegnet import GhanaSegNet
from ghanasegnet_v2 import GhanaSegNetV2

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def analyze_model_size(model, model_name):
    """Analyze model size and memory usage"""
    total_params, trainable_params = count_parameters(model)
    
    # Calculate memory usage (approximate)
    param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
    
    return {
        'name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_size_mb': param_size_mb,
        'model': model
    }

def detailed_parameter_analysis(model, model_name):
    """Detailed parameter breakdown by layer"""
    print(f"\n🔍 {model_name} - Detailed Parameter Breakdown")
    print("=" * 60)
    
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:<40}: {param_count:>10,}")
    
    print("-" * 60)
    print(f"{'TOTAL':<40}: {total_params:>10,}")
    return total_params

def main():
    print("🧮 PARAMETER COUNT ANALYSIS FOR FAIR COMPARISON")
    print("=" * 80)
    
    # Define models with their configurations
    models_config = [
        ('UNet', lambda: UNet(n_channels=3, n_classes=6)),
        ('DeepLabV3+', lambda: DeepLabV3Plus(num_classes=6, backbone='resnet50')),
        ('SegFormer-B0', lambda: SegFormerB0(num_classes=6)),
        ('GhanaSegNet (Original)', lambda: GhanaSegNet(num_classes=6)),
        ('GhanaSegNet v2', lambda: GhanaSegNetV2(num_classes=6)),
    ]
    
    # Analyze all models
    model_stats = []
    
    for model_name, model_factory in models_config:
        try:
            print(f"\n📊 Analyzing {model_name}...")
            model = model_factory()
            stats = analyze_model_size(model, model_name)
            model_stats.append(stats)
            print(f"✅ {model_name}: {stats['total_params']:,} parameters")
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {e}")
            continue
    
    # Sort by parameter count
    model_stats.sort(key=lambda x: x['total_params'])
    
    # Display comparison table
    print(f"\n📋 PARAMETER COUNT COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<25} {'Total Params':<15} {'Trainable':<15} {'Size (MB)':<12}")
    print("-" * 80)
    
    for stats in model_stats:
        print(f"{stats['name']:<25} {stats['total_params']:<15,} {stats['trainable_params']:<15,} {stats['param_size_mb']:<12.1f}")
    
    # Find target parameter count for fair comparison
    baseline_params = [stats['total_params'] for stats in model_stats if 'GhanaSegNet' not in stats['name']]
    target_params = sum(baseline_params) // len(baseline_params)  # Average
    
    print(f"\n🎯 FAIR COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Baseline models parameter range: {min(baseline_params):,} - {max(baseline_params):,}")
    print(f"Average baseline parameters: {target_params:,}")
    
    # Find GhanaSegNet v2 stats
    v2_stats = next((stats for stats in model_stats if 'v2' in stats['name']), None)
    if v2_stats:
        current_v2_params = v2_stats['total_params']
        print(f"Current GhanaSegNet v2 parameters: {current_v2_params:,}")
        
        if current_v2_params > max(baseline_params):
            reduction_needed = current_v2_params - target_params
            reduction_percent = (reduction_needed / current_v2_params) * 100
            print(f"⚠️  v2 needs reduction: {reduction_needed:,} parameters ({reduction_percent:.1f}%)")
            
            # Suggest optimization strategies
            print(f"\n💡 OPTIMIZATION STRATEGIES FOR v2:")
            print(f"   1. Reduce EfficientNet backbone size (e.g., EfficientNet-B0 instead of B2)")
            print(f"   2. Reduce transformer embedding dimensions")
            print(f"   3. Use depthwise separable convolutions in cultural attention")
            print(f"   4. Reduce decoder channel dimensions")
            print(f"   5. Use shared weights in multi-scale fusion")
            
        elif current_v2_params < min(baseline_params):
            print(f"✅ v2 is lighter than baselines - consider fair comparison achieved")
        else:
            print(f"✅ v2 is within baseline range - fair comparison achieved")
    
    # Detailed analysis of GhanaSegNet v2
    if v2_stats:
        detailed_parameter_analysis(v2_stats['model'], 'GhanaSegNet v2')
    
    # Memory usage analysis
    print(f"\n💾 MEMORY USAGE ANALYSIS (Training)")
    print("=" * 80)
    print(f"{'Model':<25} {'Parameters (MB)':<18} {'Est. Training (GB)':<18}")
    print("-" * 80)
    
    for stats in model_stats:
        # Estimate training memory (parameters + gradients + activations + optimizer states)
        # Rule of thumb: ~4-8x parameter memory for training
        training_memory_gb = stats['param_size_mb'] * 6 / 1024  # Conservative estimate
        print(f"{stats['name']:<25} {stats['param_size_mb']:<18.1f} {training_memory_gb:<18.1f}")
    
    return model_stats, target_params

if __name__ == "__main__":
    model_stats, target_params = main()
    
    print(f"\n🏆 RECOMMENDATION:")
    print(f"Target parameter count for fair comparison: ~{target_params:,} parameters")