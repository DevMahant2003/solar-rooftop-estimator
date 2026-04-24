import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# --- 1. SETTINGS ---
DATA_DIR = "./data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")

# Auto-detect M4 GPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Device set to: {DEVICE}")


# --- 2. DATASET LOADER ---
class RooftopDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ids = os.listdir(images_dir)
        # Filter for jpg/png images only
        self.ids = [
            x for x in self.ids if x.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    def __getitem__(self, i):
        image_filename = self.ids[i]

        # Load Image
        image_path = os.path.join(self.images_dir, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find Matching Mask (Masks are often .png even if image is .jpg)
        mask_filename = image_filename.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_filename)

        # Load Mask
        mask = cv2.imread(mask_path, 0)  # 0 = Load as grayscale

        # Safety check: if mask didn't load (filename mismatch), create a blank one or error
        if mask is None:
            print(
                f"⚠️ Warning: Mask not found for {image_filename}. Expected: {mask_filename}"
            )
            # Create dummy mask to prevent crash
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Resize (Standard size for DeepLab)
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # Normalize to 0-1 and convert to Tensor
        # Image: (H, W, 3) -> (3, H, W)
        image = np.transpose(image, (2, 0, 1)).astype("float32") / 255.0
        # Mask: (H, W) -> (1, H, W)
        mask = (
            np.expand_dims(mask, axis=0).astype("float32") / 255.0
        )  # Binary mask 0.0 or 1.0

        return torch.from_numpy(image), torch.from_numpy(mask)

    def __len__(self):
        return len(self.ids)


# --- 3. MODEL & TRAINING ---
# Initialize Data
dataset = RooftopDataset(IMAGES_DIR, MASKS_DIR)

if len(dataset) == 0:
    print("❌ Error: No images found. Check your 'data/images' folder.")
    exit()

dataloader = DataLoader(
    dataset, batch_size=4, shuffle=True
)  # Low batch size for Mac Air

# Load Model
model = smp.DeepLabV3Plus(
    encoder_name="resnet18",  # Lighter backbone for faster training on Mac
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid",
)
model.to(DEVICE)

loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("🚀 Starting Training on M4...")
epochs = 30
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for images, masks in dataloader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / len(dataloader):.4f}")

# --- 4. TEST RESULT ---
print("✅ Training Done. Showing a sample result...")
model.eval()
image, mask = dataset[0]  # Take first image
with torch.no_grad():
    pred = model(image.unsqueeze(0).to(DEVICE))
    pred = pred.cpu().numpy()[0, 0]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
plt.subplot(1, 3, 2)
plt.title("True Mask")
plt.imshow(mask.numpy()[0], cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Predicted")
plt.imshow(pred, cmap="gray")
plt.show()

# ... (Training loop matches above) ...

# --- 4. SAVE & TEST ---
print("💾 Saving model to rooftop_model.pth...")
torch.save(model.state_dict(), 'rooftop_model.pth')  # <--- THIS IS THE MISSING LINE

print("✅ Training Done. Showing a sample result...")
model.eval()
image, mask = dataset[0] 
with torch.no_grad():
    pred = model(image.unsqueeze(0).to(DEVICE))
    pred = pred.cpu().numpy()[0, 0]

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
plt.subplot(1, 3, 2); plt.title("True Mask"); plt.imshow(mask.numpy()[0], cmap='gray')
plt.subplot(1, 3, 3); plt.title("Predicted"); plt.imshow(pred, cmap='gray')
plt.show()
