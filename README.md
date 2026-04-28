# SMS Spam Classification using SVM

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using a Support Vector Machine (SVM) model. This project was developed through a research-driven approach, analyzing 16 research papers to identify SVM as the most accurate model for this task.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Research Background](#research-background)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Project Structure](#project-structure)
- [Team](#team)

---

## Project Overview

This project builds an SMS spam detection system using Natural Language Processing (NLP) and supervised machine learning. Messages are vectorized using TF-IDF and classified using a linear-kernel SVM — a model chosen after a thorough review of existing literature.

---

## Research Background

This project is backed by a structured research phase conducted by a team of 3 members. After reviewing **16 research papers** on SMS spam detection and text classification, the **Support Vector Machine (SVM)** was selected as the primary model due to its consistently superior performance in high-dimensional, sparse text classification tasks.

Key findings from the literature review include:
- SVM with a linear kernel outperforms Naive Bayes, Decision Trees, and Logistic Regression in SMS classification scenarios.
- TF-IDF vectorization combined with SVM yields the best balance of precision and recall.
- Linear SVM is robust against overfitting on small-to-medium text datasets.

---

## Dataset

**Dataset:** [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

| Property       | Details                          |
|----------------|----------------------------------|
| File name      | `SMSSpamCollection`              |
| Format         | Tab-separated (no header)        |
| Total messages | 5,572                            |
| Labels         | `ham` (legitimate), `spam`       |
| Encoding       | UTF-8                            |

The dataset file must be placed in the same directory as the script before running.

---

## Tech Stack

| Library        | Purpose                            |
|----------------|------------------------------------|
| `pandas`       | Data loading and preprocessing     |
| `scikit-learn` | TF-IDF vectorization, SVM, metrics |
| `matplotlib`   | Data and result visualization      |
| `seaborn`      | Confusion matrix heatmap           |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sms-spam-classification.git
cd sms-spam-classification
```

### 2. Install Dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### 3. Add the Dataset

Download the `SMSSpamCollection` file and place it in the project root directory.

---

## Usage

Run the classification script:

```bash
python spam_classifier.py
```

The script will:
1. Load and preview the dataset
2. Display a bar chart of spam vs. ham distribution
3. Train the SVM model
4. Print evaluation metrics to the console
5. Display a confusion matrix heatmap

---

## Model Pipeline

```
Raw SMS Messages
      │
      ▼
Label Encoding  (ham → 0, spam → 1)
      │
      ▼
Train/Test Split  (80% train, 20% test, random_state=42)
      │
      ▼
TF-IDF Vectorization  (removes English stop words)
      │
      ▼
SVM Classifier  (kernel='linear')
      │
      ▼
Predictions & Evaluation
```

---

## Results

The model is evaluated on four key metrics:

| Metric    | Description                                      |
|-----------|--------------------------------------------------|
| Accuracy  | Overall percentage of correctly classified messages |
| Precision | Of messages predicted as spam, how many truly are  |
| Recall    | Of all actual spam messages, how many were caught  |
| F1 Score  | Harmonic mean of precision and recall              |

Results are printed to the console after each run, along with a full classification report and a visual confusion matrix.

---

## Project Structure

```
sms-spam-classification/
│
├── spam_classifier.py       # Main script
├── SMSSpamCollection        # Dataset file (add manually)
└── README.md                # Project documentation
```

---

## Team

This project was developed by a team of **3 members** as part of an academic research initiative.

| Role              | Contribution                                      |
|-------------------|---------------------------------------------------|
| Research Lead     | Literature review of 16 research papers           |
| ML Engineer       | Model selection, training, and evaluation         |
| Data & Visualization | Preprocessing, EDA, and results visualization |

---

## License

This project is for academic and educational purposes. The SMS Spam Collection dataset is publicly available via the UCI Machine Learning Repository.
