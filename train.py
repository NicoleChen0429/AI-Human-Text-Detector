import re
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack

# ----------------------------
# 手工特徵
# ----------------------------
def basic_features(text: str) -> np.ndarray:
    t = text.strip()
    n_chars = len(t)
    n_words = len(re.findall(r"\w+", t))
    n_sents = max(1, len(re.findall(r"[。！？!?\.]+", t)))
    punct = len(re.findall(r"[，,。！？!?；;:：、\-\(\)（）\"'“”]", t))
    digits = len(re.findall(r"\d", t))
    # type-token ratio (粗略)
    tokens = re.findall(r"\w+", t.lower())
    ttr = (len(set(tokens)) / max(1, len(tokens))) if tokens else 0.0
    avg_sent_len = n_words / max(1, n_sents)
    punct_ratio = punct / max(1, n_chars)
    digit_ratio = digits / max(1, n_chars)

    return np.array([n_chars, n_words, n_sents, ttr, avg_sent_len, punct_ratio, digit_ratio], dtype=float)

class HandCraftFeats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        feats = np.vstack([basic_features(x) for x in X])
        return feats

# ----------------------------
# 讀資料
# ----------------------------
df = pd.read_csv("data/data.csv")
df = df.dropna(subset=["text", "label"])
X = df["text"].astype(str).tolist()
y = df["label"].astype(int).values

do_stratify = (len(set(y)) == 2) and (min(np.bincount(y)) >= 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_features=60000
)

Xtr_tfidf = vectorizer.fit_transform(X_train)
Xte_tfidf = vectorizer.transform(X_test)

# 手工特徵
hand = HandCraftFeats()
Xtr_hand = hand.fit_transform(X_train)
Xte_hand = hand.transform(X_test)

# 合併特徵
Xtr = hstack([Xtr_tfidf, Xtr_hand])
Xte = hstack([Xte_tfidf, Xte_hand])

# 模型
clf = LogisticRegression(max_iter=2000)
clf.fit(Xtr, y_train)

# 評估
proba = clf.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)
print(classification_report(y_test, pred))
print("AUC:", roc_auc_score(y_test, proba))

# 輸出
dump(vectorizer, "artifacts/vectorizer.joblib")
dump(clf, "artifacts/model.joblib")
print("Saved to artifacts/")
