#!/usr/bin/env python3
"""
Clean Benchmarking Script for GhanaSegNet Research
Ensures fair comparison by running all models with consistent settings

This script:
1. Verifies clean state (no existing results)
2. Runs all 5 models with consistent random seeds
3. Saves results in standardized format
4. Provides comprehensive comparison

Author: EricBaidoo
Date: September 26, 2025
"""

import subprocess
import os
import json
from datetime import datetime

def verify_clean_state():
    """Verify that all model directories are clean"""
    print("🔍 VERIFYING CLEAN STATE")
    print("="*30)
    
    models = ['unet', 'deeplabv3plus', 'segformer', 'ghanasegnet', 'ghanasegnet_v2']
    all_clean = True
    
    for model in models:
        checkpoint_dir = f"checkpoints/{model}"
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            if files:
                print(f"⚠️  {model.upper()}: Has {len(files)} files - {files}")
                all_clean = False
            else:
                print(f"✅ {model.upper()}: Clean")
        else:
            print(f"📁 {model.upper()}: Directory will be created")
    
    return all_clean

def run_clean_benchmarking():
    """Run complete benchmarking with all models"""
    print("\n🚀 STARTING CLEAN BENCHMARKING")
    print("="*40)
    
    # Configuration for fair comparison
    config = {
        'epochs': 15,
        'batch_size': 8,  # Optimal batch size from analysis
        'benchmark_mode': True,  # Deterministic operations
        'timestamp': datetime.now().isoformat()
    }
    
    print("📋 BENCHMARKING CONFIGURATION:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Run training for all models
    cmd = [
        'python', 'scripts/train_baselines.py',
        '--model', 'all',
        '--epochs', str(config['epochs']),
        '--batch-size', str(config['batch_size']),
        '--benchmark-mode'  # Ensures deterministic results
    ]
    
    print("🏃 RUNNING TRAINING COMMAND:")
    print(f"   {' '.join(cmd)}")
    print()
    print("⏱️  ESTIMATED TIME: ~5-6 hours total")
    print("   (~1.1 hours per model × 5 models)")
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ BENCHMARKING COMPLETED SUCCESSFULLY!")
            return True
        else:
            print(f"\n❌ BENCHMARKING FAILED (exit code: {result.returncode})")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  BENCHMARKING INTERRUPTED BY USER")
        return False
    except Exception as e:
        print(f"\n❌ BENCHMARKING ERROR: {e}")
        return False

def show_final_results():
    """Show final benchmarking results"""
    print("\n📊 FINAL BENCHMARKING RESULTS")
    print("="*35)
    
    models = ['unet', 'deeplabv3plus', 'segformer', 'ghanasegnet', 'ghanasegnet_v2']
    results = {}
    
    for model in models:
        results_file = f"checkpoints/{model}/{model}_results.json"
        if os.path.exists(results_file):
            try:
                with open(results_file) as f:
                    data = json.load(f)
                results[model] = data
                epochs = data.get('final_epoch', 0)
                iou = data.get('best_iou', 0) * 100
                params = data.get('total_parameters', 0) / 1e6
                print(f"✅ {model.upper()}: {iou:.2f}% mIoU | {params:.1f}M params | {epochs} epochs")
            except Exception as e:
                print(f"❌ {model.upper()}: Error reading results - {e}")
        else:
            print(f"⚠️  {model.upper()}: No results file found")
    
    if results:
        print(f"\n🏆 PERFORMANCE RANKING:")
        sorted_models = sorted(results.items(), key=lambda x: x[1].get('best_iou', 0), reverse=True)
        for i, (model, data) in enumerate(sorted_models, 1):
            iou = data.get('best_iou', 0) * 100
            print(f"   {i}. {model.upper()}: {iou:.2f}% mIoU")
        
        print(f"\n📈 READY FOR COLAB RESULTS ANALYSIS!")
        print("   All models trained with consistent settings")
        print("   Results saved in standardized format")
        print("   Fair comparison achieved")

def main():
    """Main benchmarking workflow"""
    print("🎯 GHANASEGNET CLEAN BENCHMARKING")
    print("="*50)
    print("This script will train all 5 models from scratch")
    print("with consistent settings for fair comparison.")
    print()
    
    # Step 1: Verify clean state
    if not verify_clean_state():
        print("\n⚠️  Warning: Some models have existing results")
        response = input("Continue anyway? (y/N): ").lower().strip()
        if response != 'y':
            print("Benchmarking cancelled. Run clean_results.py first.")
            return
    
    # Step 2: Confirm start
    print(f"\n⏱️  This will take approximately 5-6 hours to complete.")
    response = input("Start benchmarking now? (y/N): ").lower().strip()
    if response != 'y':
        print("Benchmarking cancelled.")
        return
    
    # Step 3: Run benchmarking
    success = run_clean_benchmarking()
    
    # Step 4: Show results
    if success:
        show_final_results()
    else:
        print("\n💡 If training was interrupted, you can resume with:")
        print("   python scripts/train_baselines.py --model <model_name> --epochs 15")

if __name__ == "__main__":
    main()