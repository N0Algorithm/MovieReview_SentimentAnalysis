
import pandas as pd 
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))


import nltk
nltk.download('stopwords')


st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 3em;
        color: #4CAF50;
    }
    .review-box {
        border: 2px solid #4CAF50;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-positive {
        color: green;
        font-size: 1.5em;
        text-align: center;
    }
    .result-negative {
        color: red;
        font-size: 1.5em;
        text-align: center;
    }
    .emoji {
        font-size: 2em;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown('<h1 class="title">Movie Review Sentiment Analysis</h1>', unsafe_allow_html=True)


st.markdown('<div class="review-box">', unsafe_allow_html=True)
review = st.text_area('Enter Movie Review', height=200)
st.markdown('</div>', unsafe_allow_html=True)

def clean_review(review):
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    
    review = review.translate(str.maketrans('', '', string.punctuation))
    cleaned_review = ' '.join(stemmer.stem(word.lower()) for word in review.split() if word.lower() not in stop_words)
    
    return cleaned_review



if st.button('Predict'):
    cleaned_review = clean_review(review)
    review_vectorized = scaler.transform([cleaned_review])
    result = model.predict(review_vectorized)
    
    if result[0] == 0:
        st.markdown('<p class="result-negative">Negative Review</p>', unsafe_allow_html=True)
        st.markdown('<p class="emoji">☹️👎</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result-positive">Positive Review</p>', unsafe_allow_html=True)
        st.markdown('<p class="emoji">😊👍</p>', unsafe_allow_html=True)