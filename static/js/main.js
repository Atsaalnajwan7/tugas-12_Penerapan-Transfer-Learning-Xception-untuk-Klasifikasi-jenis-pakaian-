/* ==================================================================================
   main.js
   - Preview gambar sebelum diprediksi
   - Validasi tipe file dan ukuran file (maks 5 MB) di sisi client
   - Menampilkan progress loading saat form dikirim untuk prediksi
================================================================================== */

document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("imageInput");
    const previewContainer = document.getElementById("previewContainer");
    const previewImage = document.getElementById("previewImage");
    const uploadForm = document.getElementById("uploadForm");
    const loadingContainer = document.getElementById("loadingContainer");
    const predictBtn = document.getElementById("predictBtn");

    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB
    const ALLOWED_TYPES = [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/bmp",
        "image/webp",
    ];

    if (imageInput) {
        imageInput.addEventListener("change", function () {
            const file = imageInput.files[0];

            if (!file) {
                previewContainer.classList.add("d-none");
                return;
            }

            // Validasi tipe file
            if (!ALLOWED_TYPES.includes(file.type)) {
                showClientAlert(
                    "File yang dipilih bukan format gambar yang didukung (JPG, JPEG, PNG, BMP, WEBP)."
                );
                imageInput.value = "";
                previewContainer.classList.add("d-none");
                return;
            }

            // Validasi ukuran file (maks 5 MB)
            if (file.size > MAX_FILE_SIZE) {
                showClientAlert("Ukuran file melebihi batas maksimum 5 MB.");
                imageInput.value = "";
                previewContainer.classList.add("d-none");
                return;
            }

            // Tampilkan preview gambar
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                previewContainer.classList.remove("d-none");
            };
            reader.readAsDataURL(file);
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener("submit", function () {
            // Tampilkan progress loading dan nonaktifkan tombol agar tidak double submit
            loadingContainer.classList.remove("d-none");
            predictBtn.setAttribute("disabled", "disabled");
            predictBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i>Memproses ...';
        });
    }

    function showClientAlert(message) {
        const alertBox = document.createElement("div");
        alertBox.className =
            "alert alert-danger alert-dismissible fade show shadow-sm mt-3";
        alertBox.role = "alert";
        alertBox.innerHTML = `
            <i class="bi bi-exclamation-triangle-fill me-2"></i>${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        uploadForm.parentElement.insertBefore(alertBox, uploadForm);
    }
});
