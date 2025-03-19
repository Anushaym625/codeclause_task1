import pickle as pk
import streamlit as st

# Load the trained model and vectorizer
model = pk.load(open('model.pkl', 'rb'))
vectorizer = pk.load(open('vectorizer.pkl', 'rb'))  # Make sure this file exists

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")

review = st.text_area('Enter Movie Review:', height=150)

if st.button('Predict'):
    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review to analyze!")
    else:
        # Transform the input review
        review_vectorized = vectorizer.transform([review])
        
        # Make prediction
        prediction = model.predict(review_vectorized)
        probability = model.predict_proba(review_vectorized)

        # Display result
        if prediction[0] == 0:
            st.error(f'❌ Negative Review 😞 (Confidence: {probability[0][0]:.2%})')
        else:
            st.success(f'✅ Positive Review 😊 (Confidence: {probability[0][1]:.2%})')

# Run with: streamlit run app.py
