import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.utils import resample
import os
import glob


def Classification_Report(y_pred, y_test):
    y_pred = np.asarray(y_pred)
    y_test = np.asarray(y_test)

    tp = np.sum((y_pred == 1) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

def tokenize(text):
    text = text.lower()
    clean = []
    for ch in text:
        if ch.isalnum():
            clean.append(ch)
        else:
            clean.append(" ")
    return "".join(clean).split()


df1 = pd.read_csv("spam.csv", encoding="latin-1", usecols=[0,1], names=["label","message"], header=0)
df2 = pd.read_csv("spam2.csv")

#df2_fixed = df2.rename(columns={"text": "message"})
#df = pd.concat([df1, df2_fixed], ignore_index=True)

df=df1

spam_words = []
ham_words = []

print("Tokenizing training Data")
for label, msg in zip(df["label"], df["message"]):
    words = tokenize(msg)
    if label == "spam":
        spam_words.extend(words)
    else:
        ham_words.extend(words)

spam_counts = Counter(spam_words)
ham_counts = Counter(ham_words)

N = 600

A = set([w for w,_ in spam_counts.most_common(N)])
B = set([w for w,_ in ham_counts.most_common(N)])

spam_indicator_words = A - B
ham_indicator_words = B - A

top_N = 200

top_spam = list(spam_indicator_words)[:top_N]
top_ham = list(ham_indicator_words)[:top_N]

print("converting training data to DataFrame")
cols_spam = {
    f"spam_{w}": df["message"].apply(lambda x: tokenize(x).count(w))
    for w in top_spam
}

cols_ham = {
    f"ham_{w}": df["message"].apply(lambda x: tokenize(x).count(w))
    for w in top_ham
}

df = pd.concat([df, pd.DataFrame(cols_spam), pd.DataFrame(cols_ham)], axis=1).copy()

print("Data Balancing")
df_major = df[df.label == "ham"]
df_minor = df[df.label == "spam"]

df_minor_up = resample(
    df_minor,
    replace=True,
    n_samples=len(df_major),
    random_state=42
)

df_balanced = pd.concat([df_major, df_minor_up]).sample(frac=1, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


X = df_balanced.drop(columns=["label", "message"])
y = df_balanced["label"].map({"ham":0, "spam":1})


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

_kernel = "linear"
print(f"Training SVM, Kernel = {_kernel}")
svm_model = SVC(kernel=_kernel)
svm_model.fit(X_train, y_train)

# y_pred = svm_model.predict(X_test)

# svm_report = Classification_Report(y_pred, y_test)

class KNN():
    def __init__(self):
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y, k=3):
        print(f"Training KNN k={k}")
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.k = k

    def euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = []
        for x in X:
            dists = self.euclidean(self.X_train, x)
            indices = np.argsort(dists)
            top_k = indices[:self.k]
            labels = self.y_train[top_k]
            values, counts = np.unique(labels, return_counts=True)
            preds.append(values[np.argmax(counts)])
        return np.array(preds)
    

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = None
        self.feature_probs = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes = np.unique(y)
        self.class_priors = {}
        self.feature_probs = {}

        for c in classes:
            X_c = X[y == c]
            self.class_priors[c] = (len(X_c) + self.alpha) / (len(X) + 2 * self.alpha)
            self.feature_probs[c] = (np.sum(X_c, axis=0) + self.alpha) / (len(X_c) + 2 * self.alpha)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = []

        for x in X:
            scores = {}
            for c in self.class_priors:
                # log space
                log_prior = np.log(self.class_priors[c])
                log_likelihood = np.sum(
                    x * np.log(self.feature_probs[c]) +
                    (1 - x) * np.log(1 - self.feature_probs[c])
                )
                scores[c] = log_prior + log_likelihood
            preds.append(max(scores, key=scores.get))

        return np.array(preds)

print("Training Naive Bayes")
nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)

# print("Training KNN")
knn = KNN()
knn.fit(X_train, y_train, k=3)


nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)


test_folder = "."
files = sorted(glob.glob(os.path.join(test_folder, "*.txt")))

for file_path in files:
    with open(file_path, "r", encoding="latin-1") as f:
        msg = f.read()

    words = tokenize(msg)

    feat = {}
    for w in top_spam:
        feat[f"spam_{w}"] = words.count(w)
    for w in top_ham:
        feat[f"ham_{w}"] = words.count(w)

    email_vector = pd.DataFrame([feat])[X.columns]

    svm_pred = svm_model.predict(email_vector)[0]
    knn_pred = knn.predict(email_vector)[0]
    nb_pred  = nb.predict(email_vector)[0]

    print(f"\nFile: {os.path.basename(file_path)}")
    print("  SVM:", "spam" if svm_pred == 1 else "ham")
    print("  KNN:", "spam" if knn_pred == 1 else "ham")
    print("  NB: ", "spam" if nb_pred == 1 else "ham")
