import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load Model, Scaler, dan Daftar Kolom
model = pickle.load(open('best_titanic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_cols = pickle.load(open('feature_columns.pkl', 'rb'))

st.title("🚢 Titanic Survival Predictor")
st.write("Masukkan data penumpang untuk melihat peluang keselamatan.")

# 2. Buat Input Form (Sesuaikan dengan fitur di generate_model.py)
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Kelas Penumpang (Pclass)", [1, 2, 3])
    age = st.slider("Umur", 0, 80, 25)
    sex = st.selectbox("Jenis Kelamin", ["male", "female"])

with col2:
    fare = st.number_input("Harga Tiket (Fare)", value=10.0)
    title = st.selectbox("Gelar (Title)", ["Mr", "Miss", "Mrs", "Master", "Rare"])
    embarked = st.selectbox("Pelabuhan (Embarked)", ["S", "C", "Q"])
    family_size = st.number_input("Jumlah Anggota Keluarga", min_value=1, step=1, value=1)

# Tambahan fitur otomatis
has_cabin = st.checkbox("Punya Nomor Kabin?")

# 3. Tombol Prediksi
if st.button("Prediksi Sekarang"):
    # Buat DataFrame mentah dari input user
    raw_input = pd.DataFrame([{
        'Pclass': pclass,
        'Age': age,
        'Fare': np.log1p(fare), # Transformasi Log
        'FamilySize': family_size,
        'Has_Cabin': 1 if has_cabin else 0,
        'Sex': sex,
        'Embarked': embarked,
        'Title': title
    }])

    # Proses One-Hot Encoding agar sama dengan saat training
    df_encoded = pd.get_dummies(raw_input)

    # REINDEX: Ini bagian paling penting! 
    # Memastikan urutan kolom sama persis dengan 'feature_columns.pkl'
    # Jika ada kolom yang kurang (misal user pilih 'Mr' maka kolom 'Title_Miss' tidak ada),
    # reindex akan menambahkannya dan mengisi dengan angka 0.
    df_final = df_encoded.reindex(columns=feature_cols, fill_value=0)

    # Scaling
    input_scaled = scaler.transform(df_final)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.divider()
    if prediction == 1:
        st.success(f"### HASIL: SELAMAT! 🎉")
        st.write(f"Model memprediksi penumpang ini memiliki peluang hidup sebesar **{probability*100:.2f}%**.")
    else:
        st.error(f"### HASIL: TEWAS 💀")
        st.write(f"Model memprediksi penumpang ini memiliki peluang hidup sebesar **{probability*100:.2f}%**.")