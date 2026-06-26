import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download("stopwords")
nltk.download("wordnet")

# -----------------------------
# Load Model
# -----------------------------

model = pickle.load(open("model.pkl", "rb"))

tfidf = pickle.load(open("tfidf.pkl", "rb"))

# -----------------------------
# Text Cleaning
# -----------------------------

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📱",
    layout="centered"
)

st.title("📱 SMS Spam Detection")

st.write(
    "Enter an SMS message below to check whether it is Spam or Ham."
)

sms = st.text_area(
    "Message",
    height=150
)

if st.button("Predict"):

    if sms.strip() == "":

        st.warning("Please enter a message.")

    else:

        cleaned = clean_text(sms)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)

        if prediction[0] == 1:

            st.error("🚫 Spam Message")

        else:

            st.success("✅ Ham Message")
