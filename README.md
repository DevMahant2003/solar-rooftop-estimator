# ☀️ Solarऊर्जा Estimator: Deep Learning-Driven Rooftop Solar Potential

An automated pipeline for assessing rooftop solar energy potential using a Deep Learning-driven building footprint extraction model. This project identifies suitable rooftops from high-resolution satellite imagery and calculates estimated annual solar energy generation.

## 📖 Overview

The growing demand for renewable energy has intensified the focus on harnessing solar power in urban areas. Despite vast potential, the lack of accurate, scalable mapping tools hinders widespread adoption. This project solves that by using a custom **Attention-Guided U-Net** to segment rooftops and applying physical scale calculations to estimate panel capacity and annual energy production.

## ✨ Key Features

* **Deep Learning Segmentation:** Accurately extracts building footprints from complex urban satellite imagery.
* **Attention Mechanisms:** Utilizes Attention Gates in the decoder to focus on relevant rooftop features while ignoring background noise.
* **Scale-to-Physical Mapping:** Converts pixel area to real-world measurements ($m^2$ and $ft^2$) based on user-defined spatial scale.
* **Energy & Capacity Forecasting:** Calculates the maximum number of installable solar panels and estimates annual energy yield (kWh).
* **Interactive Web App:** A user-friendly interface ("Solarऊर्जा Estimator") to upload images, set parameters, and view instant analysis.

---

## 📊 Dataset & Preprocessing

The model was trained on a robust dataset compiled from multiple high-resolution sources.

* **Data Sources:** Google Earth (1-meter spatial resolution), Microsoft Planetary Computer, and Roboflow (COCO annotations).
* **Dataset Split:** * Training: 1,494 images
    * Validation: 280 images
    * Testing: 221 images
    * **Total:** 1,995 images
* **Preprocessing Pipeline:**
    * **Augmentation:** Applied flipping, rotating, and contrast adjustments to improve model robustness.
    * **Standardization:** Images were resized to **128x128**, normalized, and annotations were converted into binary masks.

---

## 🧠 Model Architecture

The segmentation engine is built on a modified **U-Net Architecture** designed specifically for satellite imagery.

* **Encoder (Feature Extractor):** **ResNet-34** backbone, pre-trained on ImageNet.
* **Skip Connections:** Used to transfer high-resolution spatial information from the encoder directly to the decoder.
* **Decoder (Upsampling):** Features **Attention Gates** to actively suppress irrelevant regions in the input image while highlighting salient rooftop features useful for a specific task.

---

## 📈 Training Insights & Ablation Study

Extensive experimentation was conducted to find the optimal training parameters to prevent overfitting while maximizing footprint extraction accuracy.

### **Hyperparameter Tuning**
* **Image Resolution:** Trained on 128x128 preprocessed images.
* **Epoch Experimentation:**
    * *10 Epochs:* Tested on a sample dataset (Underfit).
    * *100 Epochs:* Trained on the full dataset (Resulted in model overfitting).
    * **40 Epochs:** The sweet spot. Maintained high accuracy without memorizing the training data.
* **Regularization:** Implemented **Early Stopping** with a patience of 10 epochs to halt training when validation metrics stopped improving.

### **Performance Metrics**
| Phase | Accuracy | Loss |
| :--- | :--- | :--- |
| **Training** | 99.77% | 0.53 |
| **Testing** | 95.75% | 32.79 |

*Note: The model was thoroughly evaluated by predicting masks on unseen test data and comparing them directly against ground truth masks to ensure high real-world applicability.*

---

## 🧮 Energy Calculation Logic

Once the rooftop footprint is successfully extracted, the application runs the following deterministic calculations to estimate solar potential.

**1. Area Calculation:**
The model estimates the rooftop area based on the image size and the scale of measurement.
$$\text{Area } (m^2) = \text{Area of Image in Pixels} \times \left(\frac{\text{meters}}{\text{pixel}}\right)^2$$
$$\text{Area } (ft^2) = \text{Area } (m^2) \times 10.764$$

**2. Panel Coverage Estimation:**
Assuming standard residential solar panels measuring **3.25 x 5.5 ft**.
$$\text{Total Panels} = \frac{\text{Usable Rooftop Area } (ft^2)}{\text{Area of One Panel}}$$

**3. Solar Capacity:**
Each panel is estimated to produce a peak energy of 325 Watts (0.325 kWp).
$$\text{Total Capacity (kW)} = \text{Total Panels} \times 0.325$$

**4. Annual Energy Generation:**
$$\text{Annual Energy (kWh)} = \text{Total Capacity} \times \text{Derate Factor} \times \text{Equivalent Sun Hours}$$

---
