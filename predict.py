import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
MODEL_PATH = "rooftop_model.pth"
IMAGE_PATH = "test_image.png"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"✅ Using Device: {DEVICE}")


# --- 1. LOAD THE MODEL (CORRECTED ARCHITECTURE) ---
def load_model_safe(path):
    print("🔹 Loading as State Dictionary (Architecture: ResNet18, Classes: 1)...")

    # FIX: Changed from resnet34 to resnet18, and classes from 2 to 1
    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid",  # Essential for 1-class output
    )

    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        exit()

    return model


model = load_model_safe(MODEL_PATH)
model.to(DEVICE)
model.eval()


# --- 2. PREPROCESS IMAGE ---
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Error: Could not read image at {img_path}")
        exit()

    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to 256x256
    input_image = cv2.resize(original_image, (256, 256))

    # Normalize
    input_tensor = np.transpose(input_image, (2, 0, 1)).astype("float32") / 255.0
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)

    return original_image, input_tensor


# --- 3. PREDICT (Single Class Logic) ---
original, tensor = preprocess_image(IMAGE_PATH)

with torch.no_grad():
    tensor = tensor.to(DEVICE)
    output = model(tensor)

    # Logic for 1 Class:
    # Output is a probability map (0.0 to 1.0). We threshold at 0.5.
    mask = output.cpu().numpy()[0, 0]
    binary_mask = (mask > 0.5).astype(np.uint8)

# --- 4. VISUALIZE ---
# Resize mask back to original image size
mask_resized = cv2.resize(
    binary_mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST
)

# Create Red Overlay
red_overlay = np.zeros_like(original)
red_overlay[:, :, 0] = 255  # Red channel

# Apply mask
masked_overlay = cv2.bitwise_and(red_overlay, red_overlay, mask=mask_resized)
final_result = cv2.addWeighted(original, 1.0, masked_overlay, 0.5, 0)

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(original)
plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(mask_resized, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(final_result)
plt.show()
