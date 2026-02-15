import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("plots", exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Load original dataset
df = pd.read_csv("bbc_news_text_complexity_summarization.csv")

# Keep only sports and politics
df_filtered = df[df["labels"].isin(["sport", "politics"])]

# Keep only required columns
data = df_filtered[["text", "labels"]]

# Print summary
print("Dataset cleaned successfully")
print(data["labels"].value_counts())
print(data)
# Load dataset

X = data["text"]
y = data["labels"]

plt.figure()
df_filtered["labels"].value_counts().plot(kind="bar")
plt.xlabel("Class")
plt.ylabel("Number of Articles")
plt.title("Class Distribution of Sports and Politics Articles")
plt.tight_layout()
plt.savefig("plots/class_distribution.png")
plt.close()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED,
)

# Feature extraction
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000,random_state=SEED),
    "SVM": LinearSVC(random_state=SEED)
}

accuracies = {}
# Training and evaluation
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc

    cm = confusion_matrix(y_test, preds, labels=["sport", "politics"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["sport", "politics"]
    )
    disp.plot()
    plt.title(f"Confusion Matrix - {name.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(f"plots/confusion_matrix_{name.lower()}.png")
    plt.close()

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Models")
plt.ylim(0.9, 1.01)
plt.tight_layout()
plt.savefig("plots/model_accuracy_comparison.png")
plt.close()