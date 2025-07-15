import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from utils.losses import CombinedLoss
from utils.metrics import compute_iou, compute_pixel_accuracy
from datasets.dataset_loader import GhanaFoodDataset
from models.ghanasegnet import GhanaSegNet  # or UNet, DeepLabV3Plus, SegFormerB0

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30):
    best_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        total_iou, total_acc = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                total_iou += compute_iou(preds, masks, num_classes=6)
                total_acc += compute_pixel_accuracy(preds, masks)

        avg_iou = total_iou / len(val_loader)
        avg_acc = total_acc / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Val IoU: {avg_iou:.4f}, Acc: {avg_acc:.4f}")

        # Save best model
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print("✅ Best model saved.")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_dataset = GhanaFoodDataset(split='train')
    val_dataset = GhanaFoodDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Model
    model = GhanaSegNet(num_classes=6).to(device)

    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.8)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    os.makedirs('checkpoints', exist_ok=True)
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30)
