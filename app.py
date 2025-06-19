# app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Eksplorasi Review Ponsel", layout="wide")

# --- Fungsi untuk Memuat Data (Menggunakan Cache untuk Performa) ---
@st.cache_data
def load_data():
    """
    Fungsi ini memuat data review yang sudah dianalisis dari file PKL.
    """
    file_path = "analyzed_reviews.pkl"
    try:
        data = pd.read_pickle(file_path)
        # Mengubah kolom Brand Name menjadi string
        data['Brand Name'] = data['Brand Name'].astype(str)
        return data
    except FileNotFoundError:
        st.error(f"File '{file_path}' tidak ditemukan. Pastikan Anda telah menjalankan notebook analisis dan meletakkan file output di folder yang sama dengan app.py.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memuat data: {e}")
        return None

# --- Antarmuka Aplikasi ---

# Judul Utama
st.title("📊 Dasbor Analisis Review Ponsel")
st.write("Analisis sentimen dan aspek yang paling sering dibicarakan untuk berbagai merek ponsel.")

# Memuat data
df = load_data()

# Hanya lanjutkan jika data berhasil dimuat
if df is not None:
    # --- Input Pengguna: Kotak Pencarian Teks untuk Merek ---
    search_query = st.text_input(
        "Ketik merek HP yang ingin Anda cari:",
        placeholder="Contoh: Samsung, Apple, Xiaomi..."
    )

    st.write("---")

    # --- Tampilkan Hasil Setelah Pengguna Mencari ---
    if search_query:
        # Filter dataframe berdasarkan input pencarian (case-insensitive)
        brand_df = df[df['Brand Name'].str.contains(search_query, case=False, na=False)]

        if not brand_df.empty:
            # Dapatkan nama merek unik dari hasil filter untuk ditampilkan di judul
            brand_name_found = brand_df['Brand Name'].unique()[0]
            st.header(f"Hasil Analisis untuk: {brand_name_found}")

            # --- Tampilkan Metrik dan Diagram Pai dalam dua kolom ---
            col1, col2 = st.columns([1, 1]) # Beri rasio agar seimbang

            # --- KOLOM 1: Metrik Angka & Daftar Aspek ---
            with col1:
                st.subheader(f"Ringkasan Sentimen")
                sentiment_counts = brand_df['Sentiment_Label'].value_counts()
                positive_count = sentiment_counts.get('Positif', 0)
                negative_count = sentiment_counts.get('Negatif', 0)

                st.metric(label="👍 Jumlah Review Positif", value=int(positive_count))
                st.metric(label="👎 Jumlah Review Negatif", value=int(negative_count))
                
                # Filter untuk mendapatkan review yang memiliki aspek
                reviews_with_aspects = brand_df.dropna(subset=['Aspect'])

                if not reviews_with_aspects.empty:
                    st.subheader("Aspek yang Paling Sering Dibahas")
                    aspect_list = reviews_with_aspects['Aspect'].unique()
                    st.markdown(f"<div style='display: flex; flex-wrap: wrap; gap: 8px;'>{''.join([f'<span style=\"background-color: #e0e0e0; border-radius: 16px; padding: 4px 12px;\">{aspect}</span>' for aspect in aspect_list])}</div>", unsafe_allow_html=True)
                
            # --- KOLOM 2: DIAGRAM PAI untuk Distribusi Sentimen ---
            with col2:
                st.subheader("Distribusi Sentimen Keseluruhan")
                if positive_count > 0 or negative_count > 0:
                    labels = 'Positif', 'Negatif'
                    sizes = [positive_count, negative_count]
                    colors = ['#33FF7A', '#FF4B4B'] # Hijau untuk Positif, Merah untuk Negatif
                    
                    fig1, ax1 = plt.subplots()
                    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
                            wedgeprops={"edgecolor":"white",'linewidth': 1, 'antialiased': True})
                    ax1.axis('equal')  # Memastikan diagram pai berbentuk lingkaran.

                    st.pyplot(fig1)
                else:
                    st.info("Tidak ada data sentimen untuk ditampilkan dalam diagram.")
            
            st.write("---")
            
            # --- Tampilkan Daftar Komentar di bawah ---
            reviews_with_aspects = brand_df.dropna(subset=['Aspect'])
            if not reviews_with_aspects.empty:
                st.subheader(f"Detail Komentar (Ditemukan {len(reviews_with_aspects)} ulasan dengan aspek)")
                for index, row in reviews_with_aspects.iterrows():
                    sentiment_icon = "👍" if row['Sentiment_Label'] == 'Positif' else "👎"
                    with st.expander(f"{sentiment_icon} **Sentimen:** {row['Sentiment_Label']} | **Aspek:** {row.get('Aspect', 'N/A')}"):
                        st.write(f"**Komentar Lengkap:**")
                        st.markdown(f"> *{row['Reviews']}*")
            else:
                 st.info(f"Tidak ada review dengan aspek spesifik yang terdeteksi untuk merek yang mengandung kata '{search_query}'.")

        else:
            # Pesan jika merek yang dicari tidak ditemukan
            st.warning(f"Tidak ada review yang ditemukan untuk merek yang mengandung kata '{search_query}'. Coba kata kunci lain.")
    else:
        # Pesan awal sebelum pengguna mencari
        st.info("Silakan masukkan nama merek di kotak pencarian di atas untuk memulai analisis.")
else:
    st.warning("Aplikasi tidak dapat berjalan karena data tidak berhasil dimuat.")