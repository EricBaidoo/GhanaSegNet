from datasets.dataset_loader import GhanaFoodDataset
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Load dataset from 'data/train'
dataset = GhanaFoodDataset(root_dir='data/train')

# Get the first image and its mask
image, mask = dataset[0]

# Plot the image and mask side by side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Sample Image")
plt.imshow(image.permute(1, 2, 0))  # Convert CHW to HWC for matplotlib
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Corresponding Mask")
plt.imshow(mask, cmap='gray')  # Show grayscale mask
plt.axis('off')

plt.show()
