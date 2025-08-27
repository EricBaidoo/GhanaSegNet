import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from utils.losses import CombinedLoss
from utils.metrics import compute_iou, compute_pixel_accuracy

# Custom dataset for segmentation
from PIL import Image
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        # Only use .jpg images that have a corresponding mask
        all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.images = []
        for img_name in sorted(all_images):
            img_stem = os.path.splitext(img_name)[0]
            mask_name = img_stem + '_mask.png'
            mask_path = os.path.join(masks_dir, mask_name)
            if os.path.exists(mask_path):
                self.images.append(img_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = img_stem + '_mask.png'
        mask_path = os.path.join(self.masks_dir, mask_name)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        import torchvision.transforms as T
        resize = T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST)
        num_classes = 6  # Make sure this matches your model
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = resize(image)
            mask = resize(mask)
            image = T.ToTensor()(image)
            mask = T.PILToTensor()(mask).long()
            # Ensure mask is always [H, W]
            if mask.dim() == 3 and mask.size(0) == 1:
                mask = mask.squeeze(0)
            elif mask.dim() == 3:
                mask = mask.squeeze()
            mask = mask.clamp(0, num_classes-1)
        return image, mask
from models.unet import UNet

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


    # Dataset paths
    train_images = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', 'images')
    train_masks = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', 'grayscale_masks')
    val_images = os.path.join(os.path.dirname(__file__), '..', 'data', 'val', 'images')
    val_masks = os.path.join(os.path.dirname(__file__), '..', 'data', 'val', 'grayscale_masks')

    train_dataset = SegmentationDataset(train_images, train_masks)
    val_dataset = SegmentationDataset(val_images, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Model
    model = UNet(in_channels=3, num_classes=6, bilinear=True).to(device)

    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.8)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    os.makedirs('checkpoints', exist_ok=True)
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30)
