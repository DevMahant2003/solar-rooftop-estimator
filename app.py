import streamlit as st
import torch
import cv2
import numpy as np
import math
from PIL import Image
import segmentation_models_pytorch as smp


MODEL_PATH = "rooftop_model.pth"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Solar Panel Constants (Standard Residential)
PANEL_WIDTH_FT = 3.25
PANEL_HEIGHT_FT = 5.5
PANEL_AREA_FT2 = PANEL_WIDTH_FT * PANEL_HEIGHT_FT  # ~17.8 sq ft
PANEL_CAPACITY_KW = 0.325  # 325 Watts per panel

st.set_page_config(page_title="SolarEstimator Pro", layout="wide", page_icon="☀️")


# --- 2. LOAD MODEL (The "Brain") ---
@st.cache_resource
def load_model():
    # We use ResNet18 and 1 Class because that matches your trained file
    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(
            f"❌ Critical Error: Could not find '{MODEL_PATH}'. Please make sure the file is in this folder."
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


model = load_model()

# --- 3. MATH HELPERS ---


def calculate_meters_per_pixel(latitude, zoom_level):
    """
    Calculates the real-world size of a pixel based on Google Maps Zoom Level.
    Formula: Cosine(Lat) * Earth_Circumference / 2^Zoom
    """
    initial_resolution = 156543.03392
    cos_lat = math.cos(latitude * math.pi / 180)
    resolution = initial_resolution * cos_lat / (2**zoom_level)
    return resolution


def predict_mask(image_np):
    """
    Runs AI + Aggressive Filtering to remove road noise.
    """
    # 1. Resize & Normalize
    img_resized = cv2.resize(image_np, (256, 256))
    input_tensor = np.transpose(img_resized, (2, 0, 1)).astype("float32") / 255.0
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)

    # 2. Inference
    with torch.no_grad():
        output = model(input_tensor)
        mask_prob = output.cpu().numpy()[0, 0]

    # --- STEP 1: High Confidence Only ---
    # We raise the bar. AI must be 75% sure it's a roof.
    binary_mask = (mask_prob > 0.75).astype(np.uint8)

    # --- STEP 2: Aggressive Scrubbing (Morphology) ---
    # Use a larger kernel (5x5) to eat away thin road lines
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Resize to original resolution
    full_mask = cv2.resize(
        binary_mask,
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # --- STEP 3: "King of the Hill" Logic ---
    # We identify all separate blobs. We assume the LARGEST blob is the real house.
    # We delete anything that is too small compared to the main house.

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        full_mask, connectivity=8
    )

    cleaned_mask = np.zeros_like(full_mask)

    # Find the area of the largest blob (skipping label 0 which is background)
    if num_labels > 1:
        max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
    else:
        max_area = 0

    # Loop through all found blobs
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # LOGIC:
        # 1. Must be bigger than 1000 pixels (Absolute size)
        # 2. OR: Must be at least 15% the size of the biggest blob.
        # This keeps the main house and large annexes, but deletes road patches.
        if area > 1000 and area > (max_area * 0.15):
            cleaned_mask[labels == i] = 1

    return cleaned_mask


# --- 4. STREAMLIT UI ---

st.title("☀️ Solar Potential Estimator")
st.markdown("### AI-Powered Rooftop Analysis")

# Layout: Left column for Inputs, Right column for Results
col_input, col_result = st.columns([1, 1.2])

with col_input:
    st.info("👇 **Step 1: Calibrate Scale**")

    # Inputs for Latitude and Zoom
    c1, c2 = st.columns(2)
    lat_input = c1.number_input("Latitude", value=21.2514, format="%.4f")
    zoom_input = c2.number_input(
        "Zoom Level (Max)", value=21.00, disabled=True, format="%.2f"
    )

    # Calculate scale instantly
    scale_m_px = calculate_meters_per_pixel(lat_input, zoom_input)
    st.success(f"📏 **Scale:** 1 Pixel = {scale_m_px:.4f} meters")

    st.divider()

    st.info("👇 **Step 2: Upload Satellite Screenshot**")
    uploaded_file = st.file_uploader(
        "Upload Image (JPG/PNG)", type=["png", "jpg", "jpeg"]
    )

    with st.expander("⚙️ Advanced Settings"):
        sun_hours = st.slider("Avg. Sun Hours/Day", 2.0, 8.0, 5.0)
        derate_factor = st.slider("Efficiency (Derate Factor)", 0.5, 0.95, 0.8)

# --- 5. EXECUTION LOGIC ---
with col_result:
    if uploaded_file is not None:
        st.subheader("📊 Analysis Results")

        # Load User Image
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_image)

        if st.button("🚀 Analyze Rooftop", type="primary", use_container_width=True):
            with st.spinner("AI is scanning for rooftops..."):
                # A. Get the mask from AI
                mask = predict_mask(img_np)

                # B. Create Red Overlay Visualization
                overlay = img_np.copy()
                overlay[mask == 1] = [255, 0, 0]  # Paint mask red
                # Blend: 70% Original, 30% Red
                final_vis = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

                st.image(
                    final_vis,
                    caption="Detected Rooftop Area (Red)",
                    use_column_width=True,
                )

                # C. Calculate Area
                total_pixels = cv2.countNonZero(mask)

                if total_pixels == 0:
                    st.warning(
                        "⚠️ No rooftop detected. Try a clearer image or lower Zoom level."
                    )
                else:
                    # Math: Pixels -> Meters -> Feet
                    area_m2 = total_pixels * (scale_m_px**2)
                    area_ft2 = area_m2 * 10.764

                    # D. Calculate Energy
                    num_panels = int(area_ft2 / PANEL_AREA_FT2)
                    system_capacity_kw = num_panels * PANEL_CAPACITY_KW
                    annual_generation_kwh = (
                        system_capacity_kw * sun_hours * 365 * derate_factor
                    )

                    # E. Display Metrics
                    st.markdown("#### 📐 Dimensions")
                    m1, m2 = st.columns(2)
                    m1.metric("Usable Area (sq ft)", f"{area_ft2:,.0f}")
                    m2.metric("Usable Area (sq m)", f"{area_m2:,.0f}")

                    st.divider()

                    st.markdown("#### ⚡ Energy & Savings")
                    e1, e2, e3 = st.columns(3)
                    e1.metric("System Size", f"{system_capacity_kw:.1f} kW")
                    e2.metric("Annual Generation", f"{annual_generation_kwh:,.0f} kWh")
                    # Assuming approx ₹4 per unit of electricity
                    e3.metric("Est. Savings", f"₹{annual_generation_kwh * 4:,.0f}/yr")

    else:
        # Placeholder when no image is uploaded
        st.write("👈 *Waiting for image upload...*")
