import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# Text Cleaning Function
# -------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------
# Load Dataset
# -------------------------------

data = pd.read_csv(
    "smsspamclassification",
    sep="\t",
    header=None,
    names=["label", "message"]
)


data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Clean Messages
data["cleaned"] = data["message"].apply(clean_text)

# -------------------------------
# TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["cleaned"])
y = data["label"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = LinearSVC()

model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("SMS Spam Classification")

st.write("Enter an SMS message below to check whether it is Spam or Ham.")

sms = st.text_area("Enter Message")

if st.button("Predict"):

    cleaned_sms = clean_text(sms)

    sms_vector = vectorizer.transform([cleaned_sms])

    prediction = model.predict(sms_vector)

    if prediction[0] == 1:
        st.error("🚫 Spam Message")
    else:
        st.success("✅ Ham Message")
