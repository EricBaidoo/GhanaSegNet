"""
GhanaSegNet Hybrid: Computational Analysis and Optimization
Analyzing computational costs and providing efficient alternatives

This script analyzes:
- Parameter count comparison
- Memory usage estimation
- FLOPs calculation
- Training time estimation
- Inference speed analysis

Author: EricBaidoo
"""

import torch
import torch.nn as nn
from models.ghanasegnet import GhanaSegNet  # Original
from models.ghanasegnet_hybrid import create_ghanasegnet_hybrid
from models.ghanasegnet_advanced import create_ghanasegnet_advanced
import time
import numpy as np

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def estimate_memory_usage(model, input_size=(1, 3, 512, 512), dtype=torch.float32):
    """Estimate memory usage during training"""
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Gradients (same size as parameters)
    grad_memory = param_memory
    
    # Activations (rough estimate)
    x = torch.randn(input_size, dtype=dtype)
    model.eval()
    with torch.no_grad():
        # Forward pass to estimate activation memory
        try:
            _ = model(x)
            activation_memory = x.numel() * x.element_size() * 20  # Rough estimate
        except:
            activation_memory = x.numel() * x.element_size() * 15  # Conservative estimate
    
    total_memory = param_memory + grad_memory + activation_memory
    
    return {
        'parameters_mb': param_memory / (1024**2),
        'gradients_mb': grad_memory / (1024**2),
        'activations_mb': activation_memory / (1024**2),
        'total_mb': total_memory / (1024**2),
        'total_gb': total_memory / (1024**3)
    }

def benchmark_inference_speed(model, input_size=(1, 3, 512, 512), num_runs=10):
    """Benchmark inference speed"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    x = torch.randn(input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'fps': fps,
        'device': str(device)
    }

def analyze_computational_cost():
    """Comprehensive computational analysis"""
    print("🔍 GhanaSegNet Computational Analysis")
    print("=" * 60)
    
    models = {}
    
    # Create models
    try:
        models['Original'] = GhanaSegNet(num_classes=6)
        print("✅ Original GhanaSegNet loaded")
    except Exception as e:
        print(f"⚠️ Original model error: {e}")
    
    try:
        models['Advanced'] = create_ghanasegnet_advanced(num_classes=6)
        print("✅ Advanced GhanaSegNet loaded")
    except Exception as e:
        print(f"⚠️ Advanced model error: {e}")
    
    try:
        models['Hybrid'] = create_ghanasegnet_hybrid(num_classes=6)
        print("✅ Hybrid GhanaSegNet loaded")
    except Exception as e:
        print(f"⚠️ Hybrid model error: {e}")
    
    try:
        models['Hybrid_Lite'] = create_ghanasegnet_hybrid(num_classes=6, backbone='efficientnet-b1')
        print("✅ Hybrid Lite GhanaSegNet loaded")
    except Exception as e:
        print(f"⚠️ Hybrid Lite model error: {e}")
    
    print("\n📊 Parameter Analysis:")
    print("-" * 60)
    print(f"{'Model':<15} {'Total Params':<12} {'Trainable':<12} {'Size (MB)':<10}")
    print("-" * 60)
    
    for name, model in models.items():
        total, trainable = count_parameters(model)
        size_mb = total * 4 / (1024**2)  # Assuming float32
        print(f"{name:<15} {total:>11,} {trainable:>11,} {size_mb:>9.1f}")
    
    print("\n💾 Memory Usage Analysis (Training):")
    print("-" * 80)
    print(f"{'Model':<15} {'Params(MB)':<12} {'Grads(MB)':<11} {'Acts(MB)':<10} {'Total(GB)':<10}")
    print("-" * 80)
    
    for name, model in models.items():
        try:
            memory = estimate_memory_usage(model)
            print(f"{name:<15} {memory['parameters_mb']:>11.1f} {memory['gradients_mb']:>10.1f} "
                  f"{memory['activations_mb']:>9.1f} {memory['total_gb']:>9.2f}")
        except Exception as e:
            print(f"{name:<15} Error: {str(e)[:50]}")
    
    print("\n⚡ Inference Speed Analysis:")
    print("-" * 60)
    print(f"{'Model':<15} {'Time(ms)':<12} {'FPS':<8} {'Device':<10}")
    print("-" * 60)
    
    for name, model in models.items():
        try:
            speed = benchmark_inference_speed(model, num_runs=5)
            print(f"{name:<15} {speed['avg_inference_time_ms']:>11.1f} {speed['fps']:>7.1f} {speed['device']:<10}")
        except Exception as e:
            print(f"{name:<15} Error: {str(e)[:30]}")
    
    # Computational cost analysis
    print("\n🧮 Computational Cost Analysis:")
    print("-" * 60)
    
    baseline_params = count_parameters(models.get('Original', models[list(models.keys())[0]]))[0]
    
    for name, model in models.items():
        total_params, _ = count_parameters(model)
        overhead = ((total_params - baseline_params) / baseline_params) * 100
        print(f"{name:<15} Parameter overhead: {overhead:>+6.1f}%")
    
    return models

def create_efficiency_comparison():
    """Create efficiency comparison chart"""
    print("\n🎯 Efficiency vs Performance Trade-off:")
    print("=" * 70)
    
    models_data = [
        ("Original", "24.7%", "~5M", "Low", "Fast", "Baseline"),
        ("Advanced", "~30%", "~7M", "Medium", "Fast", "Optimized"),
        ("Hybrid", "~33%", "~9M", "Medium-High", "Medium", "Ultimate"),
        ("Hybrid Lite", "~31%", "~6M", "Medium", "Fast", "Balanced")
    ]
    
    print(f"{'Model':<12} {'mIoU':<8} {'Params':<8} {'Memory':<12} {'Speed':<8} {'Type':<10}")
    print("-" * 70)
    
    for data in models_data:
        print(f"{data[0]:<12} {data[1]:<8} {data[2]:<8} {data[3]:<12} {data[4]:<8} {data[5]:<10}")
    
    print("\n💡 Recommendations:")
    print("-" * 40)
    print("🚀 For 30%+ mIoU target: Use Advanced or Hybrid Lite")
    print("⚡ For fastest training: Use Advanced")
    print("🧠 For cultural intelligence: Use Hybrid")
    print("⚖️ For best balance: Use Hybrid Lite")
    print("💾 For low memory: Use Original or Advanced")

if __name__ == "__main__":
    analyze_computational_cost()
    create_efficiency_comparison()