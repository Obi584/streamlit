import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt_tab')
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Text Classification Model")

def load_model(path):
    with open(path, 'rb') as file:
        model_ = pickle.load(file)
    return model_

model = load_model('text_model.pkl')

def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\W', ' ', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    return text

def tokenization(text):
  tokenized_corpus = [word_tokenize(doc) for doc in text]
  return tokenized_corpus

# Insertar texto para clasificar

text = st.text_area("Comment to classify emotion")

text = clean_text(text)
text = tokenization(text)

vectorizer = TfidfVectorizer()
text = vectorizer.fit_transform(text)


if st.button("Empotion"):
    if text.strip():
        prediction = model.predict([text])[0]
        label_names = model.classes_
        st.write(f"Classified emotion: {prediction}")
    else:
        st.warning("Please write text")