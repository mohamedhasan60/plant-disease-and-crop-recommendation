# =========================================================
#  Plant Disease Detection – Unified Training Script
#  Author : Mohamed Elgohary
# =========================================================
import os, json, numpy as np, pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ============== GPU memory growth (optional) =============
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# ===================== CONFIG ============================
BASE_DIR          = '.'          # root folder containing /rice /tomato
INIT_EPOCHS       = 30           # first phase
FINE_TUNE_EPOCHS  = 15           # second phase
BATCH_SIZE        = 32
IMG_SIZE          = (224, 224)
LEARNING_RATE     = 5e-5
MODEL_NAME        = 'plant_disease_model.keras'
FINE_TUNE         = True         # set False to skip second phase
TRAIN_DIR_FINE    = r'D:\MET\...\train'  # only used if FINE_TUNE = True
VAL_DIR_FINE      = r'D:\MET\...\val'

# ============== Data generators (phase‑1) ================
train_gen = ImageDataGenerator(
    rescale=1./255, rotation_range=40, width_shift_range=0.3, height_shift_range=0.3,
    shear_range=0.3, zoom_range=0.3, horizontal_flip=True,
    brightness_range=[0.8,1.2], fill_mode='nearest', validation_split=0.2)

def load_classes(base_dir):
    cls = []
    for sub in ['rice','tomato']:
        p = os.path.join(base_dir, sub)
        if os.path.exists(p):
            cls += [d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))]
    return cls

class_names = load_classes(BASE_DIR)
print("Found classes:", class_names)

def make_flow(root, subset):
    return train_gen.flow_from_directory(
        root, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset=subset, shuffle=True)

rice_train  = make_flow(os.path.join(BASE_DIR,'rice'),   'training')
rice_val    = make_flow(os.path.join(BASE_DIR,'rice'),   'validation')
tom_train   = make_flow(os.path.join(BASE_DIR,'tomato'), 'training')
tom_val     = make_flow(os.path.join(BASE_DIR,'tomato'), 'validation')

# ---- combine flows ----------------------------------------------------------------
class Combined(tf.keras.utils.Sequence):
    def __init__(s,g1,g2): s.g1,g1,g2; s.g2=g2; s.n=len(g1)+len(g2)
    def __len__(s): return s.n
    def __getitem__(s,i):
        g,offset = (s.g1,i) if i<len(s.g1) else (s.g2,i-len(s.g1))
        imgs,labels = g[offset]
        new = np.zeros((imgs.shape[0], len(class_names)))
        class_idx = list(g.class_indices.keys())
        mapping   = {k:class_names.index(k) for k in class_idx}
        for j,l in enumerate(labels):
            new[j, mapping[class_idx[np.argmax(l)]]] = 1
        return imgs,new
train_comb = Combined(rice_train,tom_train)
val_comb   = Combined(rice_val,  tom_val)

# ---- class weights -----------------------------------------------------------------
y_tmp=[]
for i in range(len(train_comb)):
    _,lab = train_comb[i]; y_tmp += list(np.argmax(lab,axis=1))
cw = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_tmp),
                                                      y=y_tmp)))

# =================== Build model =========================
base = MobileNet(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
for lyr in base.layers[:80]: lyr.trainable=False   # freeze early layers

x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
out= Dense(len(class_names), activation='softmax')(x)
model = Model(base.input, out)

model.compile(optimizer=Adam(LEARNING_RATE),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_'+MODEL_NAME, monitor='val_accuracy', save_best_only=True)
]

# =================== Train phase‑1 =======================
hist = model.fit(train_comb, epochs=INIT_EPOCHS,
                 validation_data=val_comb,
                 class_weight=cw, callbacks=callbacks)

model.save(MODEL_NAME)
print(f"✔ Saved initial model → {MODEL_NAME}")

# =================== Fine‑Tuning (phase‑2) ===============
if FINE_TUNE:
    model = load_model(MODEL_NAME)
    # unfreeze more layers if needed
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    aug = ImageDataGenerator(
        rescale=1./255, rotation_range=40, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, vertical_flip=True,
        brightness_range=[0.8,1.2], fill_mode='nearest')

    train_fine = aug.flow_from_directory(TRAIN_DIR_FINE, target_size=IMG_SIZE,
                                         batch_size=BATCH_SIZE, class_mode='categorical')
    val_fine   = aug.flow_from_directory(VAL_DIR_FINE,   target_size=IMG_SIZE,
                                         batch_size=BATCH_SIZE, class_mode='categorical')

    train_labels = train_fine.classes
    cw_fine = dict(enumerate(class_weight.compute_class_weight(
        'balanced', classes=np.unique(train_labels), y=train_labels)))

    fine_callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    model.fit(train_fine, epochs=FINE_TUNE_EPOCHS,
              validation_data=val_fine,
              class_weight=cw_fine, callbacks=fine_callbacks)

    model.save('plant_disease_model_updated.keras')
    print("✔ Saved fine‑tuned model → plant_disease_model_updated.keras")

# =================== Evaluation & Plots =================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='val')
plt.title('Accuracy'); plt.legend()
plt.subplot(1,2,2)
plt.plot(hist.history['loss'], label='train'); plt.plot(hist.history['val_loss'], label='val')
plt.title('Loss'); plt.legend()
plt.tight_layout(); plt.savefig('training_plots.png')

val_pred, val_true = [], []
for i in range(len(val_comb)):
    imgs, labels = val_comb[i]
    preds = model.predict(imgs, verbose=0)
    val_pred += list(np.argmax(preds, axis=1))
    val_true += list(np.argmax(labels, axis=1))

print(classification_report(val_true, val_pred, target_names=class_names))

cm = confusion_matrix(val_true, val_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout(); plt.savefig('confusion_matrix.png')
print("✔ All done – plots saved.")
