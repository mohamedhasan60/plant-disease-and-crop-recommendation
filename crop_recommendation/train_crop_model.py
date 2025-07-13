# =========================================================
#  Crop Recommendation Model  (Temperature + Humidity)
#  Author : Mohamed Elgohary
# =========================================================
import os, joblib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===================== CONFIG ============================
DATA_CSV       = 'climate_plant_dataset.csv'     # dataset path
OUT_MODEL_KERAS= 'climate_plant_model.keras'
OUT_MODEL_H5   = 'climate_plant_model.h5'
LABEL_ENCODER  = 'label_encoder.pkl'
PLOT_HISTORY   = True         # save accuracy / loss plot
EPOCHS         = 200
BATCH_SIZE     = 32
VAL_SPLIT      = 0.5
LR             = 5e-5
SEED           = 42

# ==================== Load dataset =======================
data = pd.read_csv(DATA_CSV)
X = data[['Temperature', 'Humidity']].values
y = data['Plant'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=SEED)

# ===================== Build model =======================
model = Sequential([
    Dense(256, input_dim=2, activation='relu'),
    BatchNormalization(), Dropout(0.30),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.30),
    Dense(64,  activation='relu'),
    BatchNormalization(), Dropout(0.30),
    Dense(32,  activation='relu'),
    BatchNormalization(),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer=Adam(LR),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_'+OUT_MODEL_KERAS, monitor='val_accuracy', save_best_only=True)
]

# ======================= Training ========================
hist = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🔎 Test Accuracy: {test_acc*100:.2f}%")

# =================== Save artefacts ======================
model.save(OUT_MODEL_KERAS)
model.save(OUT_MODEL_H5)
joblib.dump(le, LABEL_ENCODER)
print("✔ Models and label encoder saved.")

# =================== Plot history ========================
if PLOT_HISTORY:
    plt.figure(figsize=(10,4))
    # accuracy
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='val')
    plt.title('Accuracy'); plt.legend()
    # loss
    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.title('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig('climate_training_plots.png')
    print("📊 Training plots saved → climate_training_plots.png")
