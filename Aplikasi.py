import numpy as np
import pandas as pd
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Menampilkan gambar
image = st.image('icd_image.png', use_column_width=True, caption='Gambar ICD')

df = pd.read_csv('dataset.csv', encoding='latin1')
df = df.sample(frac=1)

vectorizer = TfidfVectorizer(stop_words="english")

X = df['Diagnosis']
Y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)  # Splitting dataset

# Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi', SelectKBest(chi2, k=13)),
                     ('clf', RandomForestClassifier(random_state=0))]  
                   )

model = pipeline.fit(X_train, y_train)

st.title("Automatic ICD Menggunakan Machine Learning")

# Menggunakan st.card untuk tampilan kartu nama
with st.card:
    st.subheader("Input Diagnosis Penyakit")
    diagnosis = st.text_area('Masukkan Diagnosis Penyakit')
    if st.button('Automatic Coding'):
        st.subheader("Hasil")
        st.write("Kode ICD = ", predict_kode(diagnosis))
        accuracy = model.score(X_test, y_test)
        st.write("Akurasi: {:.2f}%".format(accuracy * 100))
