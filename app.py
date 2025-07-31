# app.py (Versi Final - Siap Deploy)
import streamlit as st
import pickle
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Download data NLTK dan muat model spaCy ---
@st.cache_resource
def load_dependencies():
    """Memuat semua dependensi yang dibutuhkan sekali saja."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        # Menggunakan 'punkt' yang benar, bukan 'punkt_tab'
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        st.info("Model spaCy tidak ditemukan, sedang mengunduh...")
        from spacy.cli import download
        download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
        st.success("Model spaCy berhasil diunduh.")
    return nlp

nlp = load_dependencies()
stop_words = set(stopwords.words('english'))

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_models():
    """Memuat model yang diperlukan untuk prediksi sentimen."""
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            sentiment_model = pickle.load(f)
        return vectorizer, sentiment_model
    except FileNotFoundError as e:
        st.error(f"Error memuat model: {e}. Pastikan file 'vectorizer.pkl' dan 'model.pkl' ada di folder yang sama.")
        return None, None

# --- Fungsi Pra-pemrosesan Teks (Sesuai Colab) ---
def preprocess_text(text):
    """Membersihkan teks input agar sesuai dengan format data training."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)

    processed_tokens = []
    doc = nlp(" ".join(tokens))
    for token in doc:
        if token.text not in stop_words and token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']:
            processed_tokens.append(token.lemma_)
    return " ".join(processed_tokens)

# --- Konfigurasi Halaman & Memuat Model ---
st.set_page_config(page_title="Prediksi Sentimen Review", layout="centered")
vectorizer, sentiment_model = load_models()

# --- Antarmuka Aplikasi ---
st.title("🤖 Prediksi Sentimen Review")
st.write("Masukkan ulasan ponsel (dalam Bahasa Inggris) untuk memprediksi sentimennya.")
st.write("---")

# --- Fitur Contoh Input ---
st.write("**Tidak punya teks untuk dicoba? Klik contoh di bawah ini:**")
col1, col2 = st.columns(2)
positive_example = "I absolutely love this phone! The camera quality is stunning and the battery lasts all day with heavy use. Highly recommended!"
negative_example = "This was a terrible purchase. The phone started lagging after just a week and the battery life is disappointing. I regret buying it."

# Menggunakan session state untuk menangani pembaruan teks area
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

if col1.button("Coba Contoh Positif"):
    st.session_state.user_input = positive_example
    st.rerun()

if col2.button("Coba Contoh Negatif"):
    st.session_state.user_input = negative_example
    st.rerun()

user_input = st.text_area("Tulis atau salin ulasan di sini:", value=st.session_state.user_input, height=150, key="main_text_area")


if st.button("Prediksi Sentimen", key="predict_button"):
    if all([vectorizer, sentiment_model]) and user_input:

        # 1. Pra-pemrosesan teks
        with st.spinner("Menganalisis teks..."):
            processed_text_str = preprocess_text(user_input)

        # Tampilkan teks yang sudah diproses
        with st.expander("Lihat Teks yang Sudah Diproses"):
            st.info("Ini adalah teks yang dianalisis oleh model setelah dibersihkan (dihapus stopwords, simbol, dll.)")
            st.write(f"> *{processed_text_str}*")

        # 2. Prediksi Sentimen
        vectorized_input = vectorizer.transform([processed_text_str])
        prediction = sentiment_model.predict(vectorized_input)
        prediction_proba = sentiment_model.predict_proba(vectorized_input)

        # 3. Tampilkan hasil
        st.subheader("Hasil Analisis:")

        if prediction[0] == 1:
            st.metric(label="Prediksi Sentimen", value="Positif 👍")
            st.success(f"Skor Kepercayaan: {prediction_proba[0][1]:.2%}")
            st.balloons()
        else:
            st.metric(label="Prediksi Sentimen", value="Negatif 👎")
            st.error(f"Skor Kepercayaan: {prediction_proba[0][0]:.2%}")

    else:
        st.warning("Model tidak berhasil dimuat atau tidak ada input teks.")