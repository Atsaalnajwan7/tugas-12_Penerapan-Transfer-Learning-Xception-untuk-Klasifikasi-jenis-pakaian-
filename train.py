"""
==================================================================================
train.py
Tugas 12 - Praktikum Sistem Multimedia
Klasifikasi Jenis Pakaian Menggunakan Transfer Learning Xception

Script ini melakukan:
1. Membaca dataset dari folder dataset/
2. Melakukan split otomatis (train 70% / val 15% / test 15%) jika belum ada
3. Membangun model Transfer Learning berbasis Xception
4. Melatih model dengan augmentasi data
5. Menyimpan model, label kelas, history, grafik, dan evaluasi lengkap
==================================================================================
"""

import os
import shutil
import random
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # agar bisa jalan tanpa GUI (server/headless)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==================================================================================
# KONFIGURASI GLOBAL
# ==================================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")          # dataset mentah / hasil split
SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")       # folder hasil split otomatis
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_NAMES = [
    "dress", "hat", "longsleeve", "outwear", "pants",
    "shirt", "shoes", "shorts", "skirt", "t-shirt",
]

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==================================================================================
# TAHAP 1: PERSIAPAN DATASET (SPLIT OTOMATIS JIKA BELUM ADA)
# ==================================================================================
def dataset_already_split(base_path):
    """Mengecek apakah folder train/validation/test sudah ada dan berisi data."""
    required = ["train", "validation", "test"]
    for folder in required:
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            return False
        # pastikan minimal ada 1 kelas dengan file di dalamnya
        subfolders = [f for f in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, f))]
        if len(subfolders) == 0:
            return False
    return True


def collect_images_per_class(source_dir, class_names):
    """
    Mengumpulkan path gambar per kelas dari folder dataset mentah.
    Struktur yang diharapkan: dataset/<nama_kelas>/*.jpg
    """
    class_to_files = {}
    for cls in class_names:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            print(f"[PERINGATAN] Folder kelas '{cls}' tidak ditemukan di {source_dir}, dilewati.")
            continue
        files = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        if len(files) == 0:
            print(f"[PERINGATAN] Tidak ada gambar pada kelas '{cls}'.")
            continue
        class_to_files[cls] = files
    return class_to_files


def split_dataset(source_dir, split_dir, class_names):
    """
    Melakukan split dataset menjadi train (70%), validation (15%), test (15%)
    kemudian menyalin file ke struktur folder baru: split_dir/train|validation|test/<kelas>/
    """
    print(">> Folder train/validation/test belum ditemukan. Melakukan split otomatis ...")

    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)

    class_to_files = collect_images_per_class(source_dir, class_names)

    if len(class_to_files) == 0:
        raise RuntimeError(
            "Dataset tidak ditemukan atau kosong. "
            "Pastikan folder dataset/ berisi subfolder per kelas dengan gambar di dalamnya."
        )

    for cls, files in class_to_files.items():
        # split pertama: train vs (val+test)
        train_files, temp_files = train_test_split(
            files, train_size=TRAIN_RATIO, random_state=SEED, shuffle=True
        )
        # split kedua: val vs test dari sisa data
        val_relative = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_files, test_files = train_test_split(
            temp_files, train_size=val_relative, random_state=SEED, shuffle=True
        )

        for subset_name, subset_files in [
            ("train", train_files),
            ("validation", val_files),
            ("test", test_files),
        ]:
            target_dir = os.path.join(split_dir, subset_name, cls)
            os.makedirs(target_dir, exist_ok=True)
            for src_file in subset_files:
                dst_file = os.path.join(target_dir, os.path.basename(src_file))
                shutil.copyfile(src_file, dst_file)

        print(f"   Kelas '{cls}': train={len(train_files)}, "
              f"val={len(val_files)}, test={len(test_files)}")

    print(">> Split dataset selesai.\n")


def prepare_dataset():
    """
    Menentukan folder dataset yang akan dipakai untuk training.
    Jika dataset/ sudah memiliki struktur train/validation/test -> pakai langsung.
    Jika belum -> lakukan split otomatis ke dataset_split/.
    """
    if dataset_already_split(DATASET_DIR):
        print(">> Dataset sudah memiliki folder train/validation/test. Menggunakan struktur yang ada.\n")
        return (
            os.path.join(DATASET_DIR, "train"),
            os.path.join(DATASET_DIR, "validation"),
            os.path.join(DATASET_DIR, "test"),
        )
    else:
        if not os.path.isdir(DATASET_DIR):
            raise RuntimeError(
                f"Folder dataset tidak ditemukan di '{DATASET_DIR}'. "
                "Silakan letakkan dataset pada folder dataset/ dengan subfolder per kelas."
            )
        split_dataset(DATASET_DIR, SPLIT_DIR, CLASS_NAMES)
        return (
            os.path.join(SPLIT_DIR, "train"),
            os.path.join(SPLIT_DIR, "validation"),
            os.path.join(SPLIT_DIR, "test"),
        )


# ==================================================================================
# TAHAP 2: DATA GENERATOR (AUGMENTASI)
# ==================================================================================
def build_generators(train_dir, val_dir, test_dir):
    """Membuat ImageDataGenerator untuk training, validation, dan test."""

    # Training: menggunakan augmentasi lengkap
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Validation & test: hanya rescale (tidak diaugmentasi)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=SEED,
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


# ==================================================================================
# TAHAP 3: MEMBANGUN MODEL TRANSFER LEARNING XCEPTION
# ==================================================================================
def build_model(num_classes):
    """Membangun model Transfer Learning berbasis Xception."""

    # Load base model Xception tanpa top layer, dengan bobot ImageNet
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling=None,
    )

    # Freeze seluruh base model terlebih dahulu
    base_model.trainable = False

    # Tambahkan classifier head sesuai spesifikasi
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ==================================================================================
# TAHAP 4: TRAINING
# ==================================================================================
def train_model(model, train_generator, val_generator):
    """Melatih model dengan callback EarlyStopping, ReduceLROnPlateau, ModelCheckpoint."""

    checkpoint_path = os.path.join(MODEL_DIR, "model_xception.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ==================================================================================
# TAHAP 5: MENYIMPAN GRAFIK ACCURACY & LOSS
# ==================================================================================
def save_training_plots(history):
    """Membuat dan menyimpan grafik accuracy dan loss ke folder results/."""

    hist = history.history

    # --- Grafik Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(hist["accuracy"], label="Train Accuracy")
    plt.plot(hist["val_accuracy"], label="Validation Accuracy")
    plt.title("Grafik Accuracy Training vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_plot.png"), dpi=150)
    plt.close()

    # --- Grafik Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(hist["loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Validation Loss")
    plt.title("Grafik Loss Training vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_plot.png"), dpi=150)
    plt.close()

    print(">> Grafik accuracy dan loss disimpan ke folder results/.")


# ==================================================================================
# TAHAP 6: EVALUASI MODEL PADA DATA TEST
# ==================================================================================
def evaluate_model(model, test_generator):
    """Menghitung accuracy, precision, recall, f1-score, confusion matrix, classification report."""

    test_generator.reset()
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )

    # Simpan ringkasan metrik ke file teks
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== HASIL EVALUASI MODEL ===\n\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n\n")
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(report)
        f.write("\n=== CONFUSION MATRIX (raw) ===\n")
        f.write(np.array2string(cm))

    # Simpan classification report juga sebagai CSV agar mudah dibaca
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(RESULTS_DIR, "classification_report.csv")
    )

    # Simpan confusion matrix sebagai gambar
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("Label Sebenarnya")
    plt.xlabel("Label Prediksi")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Simpan confusion matrix mentah sebagai CSV
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(RESULTS_DIR, "confusion_matrix.csv")
    )

    print("\n>> HASIL EVALUASI MODEL")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"   Seluruh detail evaluasi disimpan ke folder: {RESULTS_DIR}\n")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# ==================================================================================
# TAHAP 7: MENYIMPAN MODEL, CLASS NAMES, DAN HISTORY
# ==================================================================================
def save_artifacts(model, history, class_names):
    """Menyimpan model final, daftar kelas (pkl), dan history training."""

    # Simpan model (memastikan tersimpan dalam format .keras walau checkpoint sudah menyimpan best model)
    model_path = os.path.join(MODEL_DIR, "model_xception.keras")
    model.save(model_path)
    print(f">> Model disimpan di: {model_path}")

    # Simpan class_names.pkl
    class_names_path = os.path.join(MODEL_DIR, "class_names.pkl")
    with open(class_names_path, "wb") as f:
        pickle.dump(class_names, f)
    print(f">> Daftar kelas disimpan di: {class_names_path}")

    # Simpan history training (pickle + json agar mudah dibaca ulang)
    history_pkl_path = os.path.join(RESULTS_DIR, "history.pkl")
    with open(history_pkl_path, "wb") as f:
        pickle.dump(history.history, f)

    history_json_path = os.path.join(RESULTS_DIR, "history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )

    print(f">> History training disimpan di: {history_pkl_path} dan {history_json_path}")


# ==================================================================================
# MAIN PIPELINE
# ==================================================================================
def main():
    print("==================================================================")
    print(" TRAINING MODEL - KLASIFIKASI JENIS PAKAIAN (XCEPTION)")
    print("==================================================================\n")

    # 1. Siapkan dataset (split otomatis jika perlu)
    train_dir, val_dir, test_dir = prepare_dataset()

    # 2. Buat data generator
    train_generator, val_generator, test_generator = build_generators(
        train_dir, val_dir, test_dir
    )

    num_classes = len(CLASS_NAMES)
    print(f">> Jumlah kelas: {num_classes} -> {CLASS_NAMES}\n")

    # 3. Bangun model
    model = build_model(num_classes)
    model.summary()

    # 4. Training
    print("\n>> Memulai training model ...\n")
    history = train_model(model, train_generator, val_generator)

    # 5. Simpan grafik training
    save_training_plots(history)

    # 6. Evaluasi model pada data test
    evaluate_model(model, test_generator)

    # 7. Simpan model, class names, history
    save_artifacts(model, history, CLASS_NAMES)

    print("\n==================================================================")
    print(" TRAINING SELESAI. Model siap digunakan pada aplikasi Flask.")
    print("==================================================================")


if __name__ == "__main__":
    main()
