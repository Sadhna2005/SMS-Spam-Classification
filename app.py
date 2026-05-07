import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

data = pd.read_csv("./SMSSpamCollection", sep='\t', header=None, names=['label','message'])

data['label'] = data['label'].map({'ham':0,'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(X_train)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

st.title("SMS Spam Detector")

sms = st.text_area("Enter a message")

if st.button("Predict"):
    sms_vector = vectorizer.transform([sms])
    result = model.predict(sms_vector)

    if result[0] == 1:
        st.error("Spam message")
    else:
        st.success("Ham message")
