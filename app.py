"""
==================================================================================
app.py
Tugas 12 - Praktikum Sistem Multimedia
Aplikasi Web Flask untuk Klasifikasi Jenis Pakaian menggunakan
Model Transfer Learning Xception.
==================================================================================
"""

import os
import pickle
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input

# ==================================================================================
# KONFIGURASI APLIKASI
# ==================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_xception.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config["SECRET_KEY"] = "sistem-multimedia-tugas-12-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==================================================================================
# LOAD MODEL DAN CLASS NAMES (SEKALI SAAT APLIKASI START)
# ==================================================================================
def load_model_and_classes():
    """Memuat model Keras dan daftar nama kelas dari disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{MODEL_PATH}'. "
            "Jalankan train.py terlebih dahulu untuk melatih dan menyimpan model."
        )
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"File class_names.pkl tidak ditemukan di '{CLASS_NAMES_PATH}'. "
            "Jalankan train.py terlebih dahulu."
        )

    print(">> Memuat model Xception ...")
    loaded_model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "rb") as f:
        loaded_class_names = pickle.load(f)

    print(">> Model dan daftar kelas berhasil dimuat.")
    return loaded_model, loaded_class_names


MODEL_LOADED = False
model = None
class_names = []

try:
    model, class_names = load_model_and_classes()
    MODEL_LOADED = True
except FileNotFoundError as e:
    # Aplikasi tetap bisa berjalan (misal untuk keperluan development halaman web),
    # tapi prediksi tidak akan berfungsi sampai model tersedia.
    print(f"[PERINGATAN] {e}")
    MODEL_LOADED = False


# ==================================================================================
# FUNGSI BANTUAN (HELPER)
# ==================================================================================
def allowed_file(filename):
    """Mengecek apakah ekstensi file termasuk dalam daftar yang diizinkan."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def is_valid_image(filepath):
    """Memvalidasi bahwa file yang diupload benar-benar merupakan gambar yang valid."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def preprocess_image(filepath):
    """
    Melakukan preprocessing gambar:
    - Membuka gambar dan konversi ke RGB
    - Resize menjadi 224x224
    - Preprocessing menggunakan preprocess_input() milik Xception
    - Menambahkan dimensi batch
    """
    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array = preprocess_input(img_array)

    return img_array


def predict_image(filepath):
    """Melakukan prediksi kelas pakaian dari gambar yang sudah diupload."""
    processed = preprocess_image(filepath)
    predictions = model.predict(processed, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100.0

    # Ambil top-3 prediksi untuk ditampilkan sebagai informasi tambahan
    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = [
        {
            "label": class_names[i],
            "confidence": round(float(predictions[i]) * 100.0, 2),
        }
        for i in top_indices
    ]

    return predicted_class, confidence, top_predictions


# ==================================================================================
# ROUTES
# ==================================================================================
@app.route("/", methods=["GET"])
def index():
    """Halaman utama: form upload gambar."""
    return render_template("index.html", model_loaded=MODEL_LOADED)


@app.route("/predict", methods=["POST"])
def predict():
    """Menangani proses upload gambar dan prediksi kelas pakaian."""

    if not MODEL_LOADED:
        flash("Model belum tersedia. Silakan jalankan train.py terlebih dahulu.", "danger")
        return redirect(url_for("index"))

    # Validasi: apakah ada file yang dikirim
    if "image" not in request.files:
        flash("Tidak ada file yang diupload.", "warning")
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        flash("Silakan pilih gambar terlebih dahulu.", "warning")
        return redirect(url_for("index"))

    # Validasi ekstensi file
    if not allowed_file(file.filename):
        flash(
            "Format file tidak didukung. Silakan upload gambar dengan format "
            "JPG, JPEG, PNG, BMP, atau WEBP.",
            "danger",
        )
        return redirect(url_for("index"))

    # Simpan file dengan nama unik agar tidak terjadi bentrok nama file
    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)

    # Validasi bahwa file benar-benar gambar yang valid (bukan sekadar ekstensinya)
    if not is_valid_image(filepath):
        os.remove(filepath)
        flash(
            "File yang diupload bukan gambar yang valid. Silakan coba dengan file gambar lain.",
            "danger",
        )
        return redirect(url_for("index"))

    # Lakukan prediksi
    try:
        predicted_class, confidence, top_predictions = predict_image(filepath)
    except Exception as e:
        flash(f"Terjadi kesalahan saat memproses gambar: {str(e)}", "danger")
        return redirect(url_for("index"))

    image_url = url_for("static", filename=f"uploads/{unique_filename}")

    return render_template(
        "result.html",
        predicted_class=predicted_class,
        confidence=round(confidence, 2),
        top_predictions=top_predictions,
        image_url=image_url,
    )


@app.route("/about")
def about():
    """Halaman tentang project."""
    return render_template("about.html")


@app.errorhandler(413)
def file_too_large(e):
    """Menangani error ketika ukuran file melebihi batas maksimum (5 MB)."""
    flash("Ukuran file terlalu besar. Maksimum ukuran upload adalah 5 MB.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html", model_loaded=MODEL_LOADED), 404


# ==================================================================================
# ENTRY POINT
# ==================================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
