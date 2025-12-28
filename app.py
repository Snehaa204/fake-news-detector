import streamlit as st
import pickle

# Load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector (India/Global)")
st.write("Enter any news headline or paragraph to check if it's fake.")

# Text input
user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter news text.")
    else:
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 0:
            st.error("❌ This news is **FAKE**")
        else:
            st.success("✔ This news is **REAL**")
