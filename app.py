import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. KONFIGURASI HALAMAN & THEME
# ==========================================
st.set_page_config(page_title="AI Cyberbullying Detector", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 2. FUNGSI PREPROCESSING
# ==========================================
kamus_gaul = {
    "bgt": "banget", "gk": "tidak", "ga": "tidak", "gak": "tidak", "gpp": "tidak apa-apa",
    "jg": "juga", "klo": "kalau", "dgn": "dengan", "kmrn": "kemarin", "skr": "sekarang",
    "skrg": "sekarang", "kya": "kayak", "kyk": "kayak", "ky": "kayak", "ajah": "saja", 
    "aja": "saja", "juga": "juga", "ato": "atau", "gausah": "tidak usah",
    "gaakan": "tidak akan", "knp": "kenapa", "ja": "saja", "krn": "karena", "sm": "sama",
    "tp": "tetapi", "tapi": "tetapi", "pas": "ketika", "lu": "kamu", "lo": "kamu", 
    "elo": "kamu", "gw": "saya", "gue": "saya", "brg": "barang", "ertong": "artis",
    "lg": "lagi", "bener2": "benar-benar", "dapet": "dapat", "biar": "agar", "lgi": "lagi",
    "tolol": "bodoh", "oon": "bodoh", "goblok": "bodoh", "geblek": "bodoh", 
    "bego": "bodoh", "sinting": "bodoh", "sakit jiwa": "bodoh", "stres": "bodoh",
    "anjing": "bodoh", "anjingg": "bodoh", "babi": "bodoh", "bangsat": "bodoh", 
    "bajingan": "bodoh", "bgst": "bodoh", "asu": "bodoh", "jancok": "bodoh", 
    "kontol": "bodoh", "kntl": "bodoh", "memek": "bodoh", "mmk": "bodoh", 
    "peler": "bodoh", "banci": "bodoh", "itil": "bodoh", "pantek": "bodoh", 
    "perek": "jelek", "lonte": "jelek", "pelacur": "jelek", "peler": "bodoh",
    "jelek": "jelek", "buruk": "jelek", "najis": "jelek", "sampah": "jelek", "ajg" : "bodoh"
}

def clean_and_normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<username>|@\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    normalized_words = [kamus_gaul.get(w, w) for w in words]
    return " ".join(normalized_words).strip()

# ==========================================
# 3. MODEL TRAINING
# ==========================================
@st.cache_resource
def load_and_train_model():
    df_train = pd.read_csv('dataset_gabungan.csv')
    df_train['text_clean'] = df_train['text'].apply(clean_and_normalize)
    
    X_train = df_train['text_clean']
    y_train = df_train['label']
    
    vec = TfidfVectorizer(ngram_range=(1, 1), max_features=3000, sublinear_tf=True)
    X_train_tfidf = vec.fit_transform(X_train)
    
    mdl = LogisticRegression(C=5.0, class_weight='balanced', random_state=42, max_iter=1000)
    mdl.fit(X_train_tfidf, y_train)
    
    return mdl, vec

try:
    model, vectorizer = load_and_train_model()
except Exception as e:
    st.error(f"Gagal memuat database internal: {e}")

# ==========================================
# 4. SIDEBAR 
# ==========================================
with st.sidebar:
    st.title("Hasil Pengujian Model")
    st.info("**Evaluasi pada 3.226 Baris Data:**")
    st.write("- **Logistic Regression:** 75.54% (Selected)")
    st.write("- **Naïve Bayes:** 72.91%")
    st.write("---")
    
    # DETAIL SPESIFIKASI MODEL
    with st.expander("Detail Parameter & Arsitektur AI"):
        st.markdown("""
        **Ekstraksi Fitur Teks:**
        - Metode: TF-IDF Vectorizer
        - Max Features: 3.000 Kata
        - N-Gram Range: (1, 1)
        - Sublinear TF: True
        
        **Hyperparameters Model:**
        - Algoritma: Logistic Regression
        - Regularization Strength (C): 5.0
        - Class Weight: Balanced
        - Max Iterations: 1.000
        - Random State: 42
        """)
        
    with st.expander("Detail Classification Report"):
        report_data = {
            "Kelas": ["Negative", "Positive"],
            "Precision": ["78%", "73%"],
            "Recall": ["73%", "78%"],
            "F1-Score": ["76%", "76%"]
        }
        st.dataframe(pd.DataFrame(report_data), hide_index=True)

        st.metric(
        label="Overall Accuracy",
        value="75,54%")
        
        st.caption("Data uji: 646 baris komentar (80:20 Split)")
        
    st.write("---")
    
    st.title("Sumber Data")
    st.caption("Model dilatih menggunakan gabungan dataset riset:")
    st.write("1. **Kaggle** (Cita Tiara Hani)")
    st.write("2. **HuggingFace** (aditdwi123)")
    st.write("3. **GitHub** (rizalespe)")
    st.write("4. **TikTok Dataset** (Prameswari et al.)")
    st.write("---")

# ==========================================
# 5. HALAMAN UTAMA 
# ==========================================
st.title("Sistem Deteksi Cyberbullying (Instagram & TikTok)")
st.write("Aplikasi berbasis Machine Learning untuk mengklasifikasikan komentar perundungan (cyberbullying) secara otomatis..")

tab1, tab2 = st.tabs(["Analisis Komparatif Data", "Uji Coba Deteksi Komentar"])

# TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
with tab1:
    st.header("Analisis Data Komentar (3.226 Data)")
    st.write("Melihat sebaran data dan sampel data komentar setelah proses pembersihan.")
    
    try:
        df = pd.read_csv('dataset_gabungan.csv')
        df['text_clean'] = df['text'].apply(clean_and_normalize)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Keseimbangan Kelas Data")
            fig, ax = plt.subplots(figsize=(4, 4))
            
            df['label'].value_counts().rename({ 'negative': 'Cyberbullying', 'positive': 'Aman' }).plot(
                kind='pie', autopct='%1.1f%%', colors=['#ff9999','#99ff99'], startangle=90, ax=ax
            )
            ax.set_ylabel('')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Isi Sampel Database Gabungan")
            st.dataframe(df[['text', 'label']].sample(10, random_state=42), use_container_width=True)
            
        st.write("---")
        st.subheader("Peta Kosakata Dominan (WordCloud)")
        
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Pola Kata Terbanyak pada Cyberbullying:**")
            df_neg = df[df['label'] == 'negative']
            text_neg = " ".join(df_neg['text_clean'].astype(str))
            wordcloud_neg = WordCloud(width=600, height=300, background_color='white', colormap='Reds').generate(text_neg)
            fig_neg, ax_neg = plt.subplots()
            ax_neg.imshow(wordcloud_neg, interpolation='bilinear')
            ax_neg.axis('off')
            st.pyplot(fig_neg)
            
        with col4:
            st.write("**Pola Kata Terbanyak pada Komentar Positif/Aman:**")
            df_pos = df[df['label'] == 'positive']
            text_pos = " ".join(df_pos['text_clean'].astype(str))
            wordcloud_pos = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(text_pos)
            fig_pos, ax_pos = plt.subplots()
            ax_pos.imshow(wordcloud_pos, interpolation='bilinear')
            ax_pos.axis('off')
            st.pyplot(fig_pos)
            
    except Exception as e:
        st.warning(f"Gagal me-load database gabungan: {e}")

# TAB 2: REAL-TIME INFERENCE TESTING
with tab2:
    st.header("Uji Coba Deteksi Komentar")
    st.write("Ketikkan atau tempel komentar media sosial di bawah ini untuk melihat apakah sistem mendeteksinya sebagai cyberbullying atau aman.")
    
    user_input = st.text_area("Ketik Komentar Di Sini:", placeholder="Contoh kalimat yang ingin diuji...")
    
    if st.button("Cek Komentar"):
        if user_input.strip() == "":
            st.warning("Mohon isi teks terlebih dahulu!")
        else:
            cleaned_input = clean_and_normalize(user_input)
            input_vectorized = vectorizer.transform([cleaned_input])
            prediction = model.predict(input_vectorized)[0]
            proba = model.predict_proba(input_vectorized)[0]
            confidence = max(proba) * 100
            
            st.write("---")
            st.subheader("Hasil Deteksi:")
            
            if prediction == 'negative':
                st.error(f"**TERDETEKSI CYBERBULLYING (Sentimen Negatif)**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.info(f"*Sistem Tokenizing:* \"{cleaned_input}\"")
            else:
                st.success(f"**KOMENTAR AMAN (Sentimen Positif)**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.info(f"*Sistem Tokenizing:* \"{cleaned_input}\"")