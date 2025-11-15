import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="OCR - Pengenalan Teks dari Gambar", layout="centered")

st.title("📄 Pengenalan Teks dari Gambar (OCR)")
st.write("Upload gambar, lalu aplikasi akan membaca teks dari gambar tersebut dan merapikannya menjadi paragraf.")

# --- Upload gambar ---
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to PIL image
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Convert to numpy array
    img_np = np.array(image)

    # EasyOCR Reader
    with st.spinner("Sedang memproses gambar..."):
        reader = easyocr.Reader(['en', 'id'], gpu=False)
        result = reader.readtext(img_np)

    st.subheader("📌 Hasil OCR (Paragraf Terdeteksi):")

    # --- PEMROSESAN PARAGRAF PENUH ---
    if len(result) == 0:
        st.warning("Tidak ada teks yang terdeteksi.")
    else:
        # Sort berdasarkan posisi Y lalu X
        result_sorted = sorted(
            result,
            key=lambda x: (x[0][0][1], x[0][0][0])  # (y, x)
        )

        paragraphs = []
        current_paragraph = []
        last_y = None

        line_threshold = 25         # perbedaan Y untuk mendeteksi baris baru
        paragraph_threshold = 45    # perbedaan Y untuk mendeteksi paragraf baru

        for (bbox, text, prob) in result_sorted:
            y_top = bbox[0][1]  # titik kiri atas

            if last_y is None:
                current_paragraph.append(text)
            else:
                gap = abs(y_top - last_y)

                if gap > paragraph_threshold:
                    # Paragraf baru
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = [text]

                elif gap > line_threshold:
                    # Baris baru dalam paragraf yang sama
                    current_paragraph.append(text)

                else:
                    # Lanjutan baris yang sama
                    current_paragraph[-1] += " " + text

            last_y = y_top

        # Masukkan paragraf terakhir
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        # Gabungkan paragraf dengan spasi + enter
        extracted_text = "\n\n".join(paragraphs)

        st.text_area("Teks Hasil OCR (Paragraf):", extracted_text, height=300)

    # --- VISUALISASI BOUNDING BOX ---
    st.subheader("📦 Visualisasi Deteksi Teks:")
    img_draw = img_np.copy()
    for (bbox, text, prob) in result:
        pts = np.array(bbox, dtype=int)
        cv2.polylines(img_draw, [pts], True, (0, 255, 0), 2)

    st.image(img_draw, caption="Deteksi teks pada gambar", use_column_width=True)

st.markdown("---")
st.write("Tugas Pemrograman Visual II")
