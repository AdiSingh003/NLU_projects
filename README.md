# Sports vs Politics Text Classification

This project implements a supervised machine learning system that classifies news articles into **Sports** or **Politics** categories using traditional Natural Language Processing (NLP) techniques.  
The task is part of an academic assignment focused on text classification and model comparison.

---

## 1. Problem Statement

Design a classifier that reads a text document and classifies it as **Sport** or **Politics** using machine learning techniques.  
The system must:
- Use feature representations such as **TF-IDF**, **Bag of Words**, or **n-grams**
- Compare **at least three machine learning models**
- Provide quantitative evaluation and analysis

---

## 2. Dataset Description

- **Dataset**: BBC News dataset  
- **Source file**: `https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset`
- **Classes used**:  
  - `sport`
  - `politics`
- **Total samples after filtering**: 908 articles  
  - Sports: 505  
  - Politics: 403  

The dataset is publicly available and contains no personal or sensitive information.

---

## 3. Preprocessing Steps

The following preprocessing steps are applied:
1. Filter dataset to keep only `sport` and `politics` labels
2. Select relevant columns (`text`, `labels`)
3. Convert text to lowercase (handled internally by TF-IDF)
4. Remove English stop words during vectorization
5. Use **TF-IDF with unigrams and bigrams**

No stemming or lemmatization is applied to preserve meaningful word patterns.

---

## 4. Models Used

The following machine learning models are trained and evaluated:

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Support Vector Machine (Linear SVM)**

All models are trained using the same feature representation and train-test split to ensure fair comparison.

---

## 5. Experimental Setup

- **Train-Test Split**: 80% training, 20% testing
- **Random Seed**: 42 (for reproducibility)
- **Feature Representation**: TF-IDF (unigrams + bigrams)
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 6. Results Summary

### Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| Naive Bayes | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 0.98 | 0.98 | 0.98 | 0.98 |
| SVM | 1.00 | 1.00 | 1.00 | 1.00 |

- Naive Bayes and SVM achieved **perfect classification** on the test set.
- Logistic Regression made a small number of misclassifications, primarily for politics articles.
- High performance is attributed to strong lexical separation between sports and politics articles.

---

## 7. Generated Plots

All plots are automatically saved in the `plots/` directory:

1. **Class Distribution**
   - `class_distribution.png`
2. **Confusion Matrices**
   - `confusion_matrix_naive bayes.png`
   - `confusion_matrix_logistic regression.png`
   - `confusion_matrix_svm.png`
3. **Model Accuracy Comparison**
   - `model_accuracy_comparison.png`

These plots help visualize dataset balance, classification errors, and comparative model performance.

---

## 8. How to Run the Code (Step-by-Step)

### Step 1: Clone or Download the Project
Ensure the following files are present:
- `problem4.py` (or your main Python script)
- `bbc_news_text_complexity_summarization.csv`

### Step 2: Install Required Libraries

Use Python 3.8 or higher.

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Step 3: Install Required Libraries

```bash
python problem4.py
```