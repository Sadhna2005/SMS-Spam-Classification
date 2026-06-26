from ensurepip import bootstrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


nltk.download('stopwords')
nltk.download('wordnet')

#Load Dataset

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label","message"]
)

print(df.head())

#Dataset Information
print("Dataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['label'].value_counts())

# Distribution Before Using Smote
plt.figure(figsize=(8,5))

sns.countplot(
    x='label',
    data=df
)

plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()

#Pie Chart
df['label'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(6,6)
)

plt.title("Spam vs Ham Distribution")
plt.ylabel("")

plt.show()

#Length Analysis
df['length'] = df['message'].apply(len)

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x='length',
    hue='label',
    bins=50,
    kde=True
)

plt.title("Message Length Distribution")

plt.show()

#Average Length Comparison
plt.figure(figsize=(8,5))

sns.boxplot(
    x='label',
    y='length',
    data=df
)

plt.title("Message Length Comparison")

plt.show()

#Text Cleaning
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

df['cleaned'] = df['message'].apply(clean_text)

#Spam word cloud
spam_words = " ".join(
    df[df['label']=='spam']['cleaned']
)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(spam_words)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Spam Word Cloud")

plt.show()

#Ham word cloud
ham_words = " ".join(
    df[df['label']=='ham']['cleaned']
)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(ham_words)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Ham Word Cloud")

plt.show()


#label Encoding
encoder = LabelEncoder()

df['label_num'] = encoder.fit_transform(
    df['label']
)

print(df[['label','label_num']].head())


#Tf-IDF vectorizartion
tfidf = TfidfVectorizer(
    max_features=5000
)

X = tfidf.fit_transform(
    df['cleaned']
)

y = df['label_num']


#before smote
print(pd.Series(y).value_counts())

#before smote graph
pd.Series(y).value_counts().plot(
    kind='bar'
)

plt.title("Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()


#apply smote
smote = SMOTE(
    random_state=42
)

X_smote, y_smote = smote.fit_resample(
    X,
    y
)

print(pd.Series(y_smote).value_counts())


#after smote graph
pd.Series(y_smote).value_counts().plot(
    kind='bar',
    color=['blue','orange']
)


plt.title("After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()

print(X_smote.shape)

#data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_smote,
    y_smote,
    test_size=0.20,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)


#data training(SVM)
svm = LinearSVC()

svm.fit(
    X_train,
    y_train
)

pred_svm = svm.predict(X_test)

#train Random forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

pred_rf = rf.predict(X_test)

#train XG boost
xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(
    X_train,
    y_train
)

pred_xgb = xgb.predict(X_test)

#Accuracy scores

svm_acc = accuracy_score(
    y_test,
    pred_svm
)

rf_acc = accuracy_score(
    y_test,
    pred_rf
)

xgb_acc = accuracy_score(
    y_test,
    pred_xgb
)

print("SVM Accuracy:", svm_acc)
print("RF Accuracy:", rf_acc)
print("XGB Accuracy:", xgb_acc)

#Precision recall
models = {
    "SVM": pred_svm,
    "Random Forest": pred_rf,
    "XGBoost": pred_xgb
}

for name,pred in models.items():

    print("\n",name)

    print(
        "Precision:",
        precision_score(y_test,pred)
    )

    print(
        "Recall:",
        recall_score(y_test,pred)
    )

    print(
        "F1 Score:",
        f1_score(y_test,pred)
    )

#classification report
    print(
        classification_report(
            y_test,
            pred_xgb
        )
    )

# Comparison graph
results = {
    "SVM": svm_acc,
    "Random Forest": rf_acc,
    "XGBoost": xgb_acc
}

plt.figure(figsize=(8,5))

plt.bar(
    results.keys(),
    results.values()
)

plt.title(
    "Model Accuracy Comparison"
)

plt.ylabel("Accuracy")

plt.show()


#Precision comparison graph

precisions = [
    precision_score(y_test,pred_svm),
    precision_score(y_test,pred_rf),
    precision_score(y_test,pred_xgb)
]

plt.figure(figsize=(8,5))

plt.bar(
    ["SVM","RF","XGB"],
    precisions
)

plt.title(
    "Precision Comparison"
)

plt.show()

#recall comparison graph
recalls = [
    recall_score(y_test,pred_svm),
    recall_score(y_test,pred_rf),
    recall_score(y_test,pred_xgb)
]

plt.figure(figsize=(8,5))

plt.bar(
    ["SVM","RF","XGB"],
    recalls
)

plt.title(
    "Recall Comparison"
)

plt.show()

# F1 comparison graph
f1_scores = [
    f1_score(y_test,pred_svm),
    f1_score(y_test,pred_rf),
    f1_score(y_test,pred_xgb)
]

plt.figure(figsize=(8,5))

plt.bar(
    ["SVM","RF","XGB"],
    f1_scores
)

plt.title(
    "F1 Score Comparison"
)

plt.show()



#confusion matrix


cm = confusion_matrix(
    y_test,
    pred_xgb
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    "XGBoost Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

cm_svm = confusion_matrix(y_test,pred_svm)
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    "SVM Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



cm_rf = confusion_matrix(y_test,pred_rf)
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


#False positive analysis

test_df = pd.DataFrame()

test_df['Actual'] = y_test
test_df['Predicted'] = pred_xgb

false_positive = test_df[
    (test_df['Actual']==0)
    &
    (test_df['Predicted']==1)
]

print(
    "False Positives:",
    len(false_positive)
)


