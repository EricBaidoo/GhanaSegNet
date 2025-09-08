#!/usr/bin/env python3
"""
Repository Cleanup Script for GhanaSegNet
Removes unnecessary files and organizes the repository for professional presentation
"""

import os
import shutil
from pathlib import Path

def cleanup_repository():
    """Remove unnecessary files and organize repository"""
    
    print("🧹 Starting repository cleanup...")
    
    # Files to remove (development/testing files)
    files_to_remove = [
        "3_WEEK_SPRINT_PLAN.md",
        "ARCHITECTURE_JUSTIFICATION.md", 
        "Chapters_1-3.md",
        "RESEARCH_PROPOSAL.md",
        "RESEARCH_TEAM_ANALYSIS.md",
        "TRANSFER_LEARNING_STRATEGY.md",
        "kk.ipynb",
        "baselinemodels.ipynb",
        "tt.py",
        "split_all.py",
        "split_dataset.py", 
        "test_evaluation.py",
        "test_import.py",
        "test_training_env.py",
        "setup_compute.sh",
        "data.zip",
        "dataset_train.zip",
        "Pipfile",
        "Pipfile.lock",
        "pyproject.toml"  # Keep requirements.txt instead
    ]
    
    # Directories to remove (if they exist and are not needed)
    dirs_to_remove = [
        "checkpoints",  # Training artifacts - should be in .gitignore
        "data"          # Dataset should not be in repo
    ]
    
    removed_files = []
    removed_dirs = []
    
    # Remove unnecessary files
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                removed_files.append(file)
                print(f"🗑️  Removed: {file}")
            except Exception as e:
                print(f"❌ Could not remove {file}: {e}")
    
    # Remove unnecessary directories
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                removed_dirs.append(dir_name)
                print(f"🗑️  Removed directory: {dir_name}")
            except Exception as e:
                print(f"❌ Could not remove {dir_name}: {e}")
    
    # Update .gitignore to prevent future clutter
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv
pip-log.txt
pip-delete-this-directory.txt

# PyTorch
*.pth
*.pt

# Training artifacts
checkpoints/
logs/
runs/
wandb/

# Data
data/
datasets/
*.zip
*.tar.gz

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Results
results/
outputs/
experiments/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ Updated .gitignore")
    
    # Create a clean project structure summary
    print("\n📁 Clean repository structure:")
    print("GhanaSegNet/")
    print("├── models/")
    print("│   ├── ghanasegnet.py")
    print("│   ├── unet.py") 
    print("│   ├── deeplabv3plus.py")
    print("│   └── segformer.py")
    print("├── utils/")
    print("│   ├── losses.py")
    print("│   └── metrics.py")
    print("├── scripts/")
    print("│   └── train_baselines.py")
    print("├── notebooks/")
    print("│   └── (research notebooks)")
    print("├── GhanaSegNet_Colab.ipynb")
    print("├── README.md")
    print("├── requirements.txt")
    print("├── LICENSE")
    print("└── .gitignore")
    
    print(f"\n🎉 Cleanup complete!")
    print(f"📄 Removed {len(removed_files)} files")
    print(f"📁 Removed {len(removed_dirs)} directories")
    print("✨ Repository is now clean and professional!")

if __name__ == "__main__":
    cleanup_repository()
