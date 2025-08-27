#!/bin/bash
# Minimal Compute Environment Setup for GhanaSegNet
# This script optimizes Codespaces for ML compute tasks

echo "🚀 Setting up minimal compute environment..."

# Set environment variables for ML optimization
export PYTHONPATH="/workspaces/GhanaSegNet:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"  # If GPU available
export OMP_NUM_THREADS="4"       # Match CPU cores

# Disable unnecessary services
echo "Disabling unnecessary services..."
sudo systemctl stop code-server 2>/dev/null || true
sudo systemctl disable code-server 2>/dev/null || true

# Clean up development packages we don't need for ML
echo "Cleaning unnecessary packages..."
sudo apt-get autoremove -y 2>/dev/null || true
sudo apt-get autoclean 2>/dev/null || true

# Set up minimal Python environment
echo "Setting up Python environment..."
python3 -m pip install --user --no-cache-dir pip --upgrade

# Essential packages only (already have PyTorch)
python3 -m pip install --user --no-cache-dir \
    numpy \
    pillow \
    tqdm \
    scikit-learn

echo "✅ Minimal compute environment ready!"
echo "📊 Available resources:"
echo "   CPU: $(nproc) cores"
echo "   RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "   Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo ""
echo "🎯 Ready for ML training/evaluation!"
