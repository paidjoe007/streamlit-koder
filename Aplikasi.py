import numpy as np
import pandas as pd
import re,string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import streamlit as st

df = pd.read_csv('dataset.csv', encoding = 'latin1')
df = df.sample(frac = 1)

from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer(stop_words="english")

X = df['Diagnosis']
Y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15) #Splitting dataset

# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=13)),
                     ('clf', LogisticRegression(random_state=0))])

model = pipeline.fit(X_train, y_train)

def predict_kode(txt):
    diagnosis_data = {'predict_kode':[txt]}
    diagnosis_data_df = pd.DataFrame(diagnosis_data)
    predict_diagnosis_cat = model.predict(diagnosis_data_df['predict_kode'])[0]
    return predict_diagnosis_cat


st.title("Automatic ICD Menggunakan Machine Learning")
diagnosis = st.text_area('Input Diagnosis Penyakit')
if st.button('Automatic Coding'):
    st.write("Kode ICD = ", predict_kode(diagnosis))
