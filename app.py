import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
 

# Load the trained model and vectorizer
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))  # This is actually a TfidfVectorizer, not a scaler

# Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis")
review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    # Transform the input review
    review_scale = scaler.transform([review])  # No need for .toarray()
    
    # Make prediction
    result = model.predict(review_scale)
    
    # Display result
    if result[0] == 0:
        st.write('Negative Review 😞')
    else:
        st.write('Positive Review 😊')
