# 🌱 plant-disease-and-crop-recommendation

**Modules**
1. **Crop Recommendation** – Dense NN, trained on temperature & humidity.
2. **Plant Disease Detection** – Transfer Learning (MobileNet).

## Repository Structure
plant-disease-and-crop-recommendation/
│
├─ README.md
├─ requirements.txt
│
├─ crop_recommendation/
│   ├─ train_crop_model.py
│   ├─ climate_plant_dataset.csv
│   ├─ climate_plant_model.keras      
│   ├─ label_encoder.pkl
│   └─ climate_training_plots.png
|   └─ Images
│
├─ disease_detection/
│   ├─ train_disease_model.py
│   ├─ plant_disease_model.keras    
│   ├─ training_plots.png
│   ├─ confusion_matrix.png
│   └─ class_indices.json
│
└─ esp32_code/
    └─ esp32_firmware.ino
