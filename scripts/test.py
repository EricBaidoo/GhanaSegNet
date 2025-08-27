# Testing / inference script

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from models.ghanasegnet import GhanaSegNet  # or UNet, DeepLabV3Plus, SegFormerB0
from utils.metrics import compute_iou
from data.dataset_loader import GhanaFoodDataset  # For color decoding

# Color map (same as in notebook)
CLASS_COLORS = np.array([
    [255, 255, 255],   # Background
    [255, 0, 0],       # Class 1
    [0, 255, 0],       # Class 2
    [0, 0, 255],       # Class 3
    [255, 255, 0],     # Class 4
    [255, 0, 255],     # Class 5
])

def decode_segmap(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls in range(len(CLASS_COLORS)):
        rgb[mask == cls] = CLASS_COLORS[cls]
    return rgb

def predict_image(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return image, pred

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GhanaSegNet(num_classes=6).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))

    image_path = "sample_test_image.jpg"  # Replace with your image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    original, pred_mask = predict_image(model, image_path, transform, device)
    pred_rgb = decode_segmap(pred_mask)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_rgb)
    plt.title("Predicted Segmentation")
    plt.tight_layout()
    plt.show()

    # Optionally save
    Image.fromarray(pred_rgb).save("results/test_prediction.png")
    print(" Saved: results/test_prediction.png")
