import streamlit as st
import pickle
import re
import string

# Load the saved model and vectorizer
model = pickle.load(open('models/pac_model.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

def clean_input(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

st.title("Fake News Detector")
st.write("Paste an article below to check its authenticity.")

user_input = st.text_area("Enter News Article Content Here:")

if st.button("Predict"):
    if user_input:
        cleaned_input = clean_input(user_input)
        vectorized_input = tfidf.transform([cleaned_input])
        prediction = model.predict(vectorized_input)
        
        if prediction[0] == 0:
            st.success("Verdict: REAL NEWS")
        else:
            st.error("Verdict: FAKE NEWS")
    else:
        st.warning("Please enter some text first.")