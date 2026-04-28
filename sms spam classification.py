import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Load Dataset
data = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

print("First 5 rows of dataset:\n")
print(data.head())


# Visualize Spam vs Ham distribution
plt.figure(figsize=(6,4))
sns.countplot(x=data['label'])
plt.title("Spam vs Ham Message Distribution")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()


# Convert labels to numeric
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)


# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Train SVM model
model = SVC(kernel='linear')

model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# Evaluation Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


print("\nModel Evaluation Metrics")
print("------------------------")

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))


# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:\n")
print(cm)


# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# Confusion Matrix Visualization
plt.figure(figsize=(6,4))

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for SMS Spam Detection")

plt.show()
