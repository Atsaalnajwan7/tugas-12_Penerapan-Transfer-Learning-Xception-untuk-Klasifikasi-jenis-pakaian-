# ClothVision — Klasifikasi Jenis Pakaian Menggunakan Transfer Learning Xception Berbasis Web (Flask)

Tugas 12 — Praktikum Sistem Multimedia

## Deskripsi Project

ClothVision adalah aplikasi web berbasis **Flask** yang mengklasifikasikan jenis pakaian
dari sebuah gambar menggunakan teknik **Transfer Learning** dengan arsitektur **Xception**
(pretrained ImageNet). Aplikasi ini mendukung 10 kelas pakaian:

- Dress
- Hat
- Longsleeve
- Outwear
- Pants
- Shirt
- Shoes
- Shorts
- Skirt
- T-Shirt

Model dilatih dengan base model Xception yang dibekukan (frozen), lalu ditambahkan
lapisan classifier custom (GlobalAveragePooling2D, Dropout, Dense, BatchNormalization)
untuk mengenali 10 kelas pakaian tersebut.

## Struktur Project

```
project/
│
├── app.py                 # Aplikasi Flask (web server & prediksi)
├── train.py                # Script training model Xception
├── requirements.txt         # Daftar dependency Python
├── runtime.txt              # Versi Python untuk deployment
├── Procfile                 # Konfigurasi deployment (Heroku-style)
├── .gitignore
├── README.md
│
├── model/
│   ├── model_xception.keras  # Model hasil training (dihasilkan oleh train.py)
│   └── class_names.pkl       # Daftar nama kelas (dihasilkan oleh train.py)
│
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/              # Folder penyimpanan gambar yang diupload user
│
├── results/                  # Grafik & hasil evaluasi (dihasilkan oleh train.py)
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix.csv
│   ├── classification_report.csv
│   ├── evaluation_summary.txt
│   ├── history.json
│   └── history.pkl
│
├── templates/
│   ├── index.html            # Halaman utama (upload gambar)
│   ├── result.html           # Halaman hasil prediksi
│   └── about.html            # Halaman tentang project
│
└── dataset/                  # Letakkan dataset mentah di sini (lihat bagian Dataset)
```

## Dataset

Dataset yang digunakan: **Clothing Dataset Small**
Sumber: https://www.kaggle.com/datasets/abdelrahmansoltan98/clothing-dataset-small

Unduh dataset dari Kaggle, lalu letakkan pada folder `dataset/` dengan struktur
per-kelas seperti berikut (sebelum di-split otomatis oleh `train.py`):

```
dataset/
├── dress/
│   ├── img001.jpg
│   └── ...
├── hat/
├── longsleeve/
├── outwear/
├── pants/
├── shirt/
├── shoes/
├── shorts/
├── skirt/
└── t-shirt/
```

Jika dataset sudah memiliki struktur `train/`, `validation/`, `test/` di dalam
folder `dataset/`, maka `train.py` akan langsung menggunakannya. Jika belum,
`train.py` akan otomatis melakukan split data dengan proporsi:

- Train: 70%
- Validation: 15%
- Test: 15%

Hasil split otomatis akan disimpan pada folder `dataset_split/`.

## Cara Install

1. Pastikan Python 3.11 sudah terpasang.
2. Buat virtual environment (opsional tapi disarankan):

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install seluruh dependency:

```bash
pip install -r requirements.txt
```

## Cara Training

1. Pastikan dataset sudah diletakkan pada folder `dataset/` (lihat bagian Dataset di atas).
2. Jalankan script training:

```bash
python train.py
```

Proses ini akan:
- Melakukan split dataset otomatis (jika diperlukan)
- Melatih model Transfer Learning Xception (minimal 20 epoch, dengan EarlyStopping,
  ReduceLROnPlateau, dan ModelCheckpoint)
- Menyimpan model ke `model/model_xception.keras`
- Menyimpan daftar kelas ke `model/class_names.pkl`
- Menyimpan grafik accuracy & loss ke `results/`
- Menghitung dan menyimpan evaluasi (accuracy, precision, recall, F1 score,
  confusion matrix, classification report) ke `results/`

Catatan: Proses training membutuhkan waktu cukup lama tergantung spesifikasi
perangkat (CPU/GPU) dan jumlah data.

## Cara Menjalankan Flask

Setelah model berhasil dilatih (folder `model/` sudah berisi `model_xception.keras`
dan `class_names.pkl`), jalankan aplikasi web:

```bash
python app.py
```

Aplikasi akan berjalan pada `http://127.0.0.1:5000/`. Buka alamat tersebut pada
browser, unggah gambar pakaian, lalu klik tombol **Prediksi Sekarang**.

## Cara Deploy

Project ini sudah dilengkapi dengan file konfigurasi untuk deployment (misalnya
ke platform seperti Heroku, Render, atau Railway):

- `requirements.txt` — dependency Python
- `runtime.txt` — versi Python yang digunakan (3.11.9)
- `Procfile` — perintah menjalankan aplikasi dengan Gunicorn (`web: gunicorn app:app`)
- `.gitignore` — mengecualikan file/folder yang tidak perlu di-commit

Langkah umum deploy (contoh menggunakan platform berbasis Git):

1. Inisialisasi git repository dan commit seluruh project (pastikan model sudah
   dilatih dan berada di folder `model/`, atau sediakan mekanisme download model
   terpisah jika ukurannya besar).
2. Hubungkan repository ke platform deployment pilihan Anda.
3. Set environment variable `PORT` jika diperlukan oleh platform.
4. Platform akan otomatis membaca `runtime.txt` dan `Procfile` untuk menjalankan
   aplikasi menggunakan Gunicorn.

```bash
git init
git add .
git commit -m "Initial commit - ClothVision"
git push <remote> <branch>
```

## Screenshot Website

> Catatan: Tangkapan layar berikut menggambarkan tampilan aplikasi setelah
> dijalankan secara lokal. Silakan jalankan aplikasi sesuai langkah di atas
> untuk melihat tampilan aktual pada perangkat Anda.

**Halaman Beranda (Upload Gambar)**
`[Screenshot halaman index.html — form upload, preview gambar, tombol prediksi]`

**Halaman Hasil Prediksi**
`[Screenshot halaman result.html — nama kelas, confidence, gambar yang diupload]`

**Halaman Tentang**
`[Screenshot halaman about.html — deskripsi project dan arsitektur model]`

## Teknologi yang Digunakan

- Python 3.11
- TensorFlow 2.x & Keras
- Flask
- Bootstrap 5 & Bootstrap Icons
- Transfer Learning Xception
- Pillow
- NumPy & Pandas
- Matplotlib
- Scikit-learn

## Lisensi

Project ini dibuat untuk keperluan akademik (Tugas 12 Praktikum Sistem Multimedia).
