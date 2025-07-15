# GhanaSegNet: A Hybrid Deep Learning Model for Semantic Segmentation of Ghanaian Foods

**GhanaSegNet** is a lightweight hybrid deep learning architecture designed to segment traditional Ghanaian meals. It integrates a CNN backbone (EfficientNet-lite0) with a Transformer head, and is optimized for culturally relevant datasets such as **FRANI**. The project also includes baselines like **UNet**, **DeepLabV3+**, and **SegFormer-B0** for comparative evaluation.

---

## 📁 Project Structure

GhanaSegNet/
├── models/ # UNet, DeepLabV3+, GhanaSegNet, SegFormer-B0
├── utils/ # Loss functions and evaluation metrics
├── scripts/ # Training, evaluation, and testing scripts
├── datasets/ # Custom PyTorch dataset loader (GhanaFoodDataset)
├── data/ # Place your FRANI dataset here
├── checkpoints/ # Saved model weights (.pth files)
├── results/ # Predicted masks and result images
├── notebooks/ # Jupyter notebooks for visualizing predictions
├── README.md # This file
└── .gitignore # Files/folders ignored by Git


---

## 🔧 Requirements

- Python 3.8+
- PyTorch ≥ 1.13
- torchvision
- matplotlib
- numpy
- scikit-learn
- Pillow

Install all dependencies using:

```bash
pip install -r requirements.txt

You may create this file manually or install packages as needed.

📦 Dataset Setup
Download the FRANI dataset (pixel-annotated Ghanaian food images) and organize it like this:

bash
Copy code
data/frani/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
Images: RGB format (e.g. .jpg, .png)

Masks: Grayscale with class IDs as pixel values

Make sure filenames match between images/ and masks/.

🚀 How to Train
Train any model (e.g., GhanaSegNet) using:

bash
Copy code
python scripts/train.py
Saves best model to checkpoints/best_model.pth

Automatically evaluates on the validation set

📊 How to Evaluate
Evaluate trained models on the test set:

bash
Copy code
python scripts/evaluate.py
Computes: Mean IoU, Accuracy, Per-class F1

Saves visual predictions to results/

🧪 Inference: Predict on a Single Image
bash
Copy code
python scripts/test.py
Modify image_path in the script

The predicted mask is saved to results/test_prediction.png

📈 Visualize Results
Use the Jupyter notebook to visually compare:

bash
Copy code
notebooks/results_visualization.ipynb
Original image

Ground truth mask

Predicted mask

📚 Citation
If you use GhanaSegNet in your research or thesis, please cite the original repository or acknowledge the contribution in your documentation.

🙌 Acknowledgements
FRANI Dataset Team

PyTorch Open Source Community

Authors of UNet, DeepLabV3+, SegFormer

Made with ❤️ for food segmentation research in Ghana.


---

Let me know if you’d like this saved into the ZIP or uploaded to GitHub directly.
