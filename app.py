import re
import numpy as np
import streamlit as st
from joblib import load
from scipy.sparse import hstack

st.set_page_config(page_title="AI / Human 文章偵測器", layout="centered")

@st.cache_resource
def load_artifacts():
    vectorizer = load("artifacts/vectorizer.joblib")
    model = load("artifacts/model.joblib")
    return vectorizer, model

def basic_features(text: str) -> np.ndarray:
    t = text.strip()
    n_chars = len(t)
    n_words = len(re.findall(r"\w+", t))
    n_sents = max(1, len(re.findall(r"[。！？!?\.]+", t)))
    punct = len(re.findall(r"[，,。！？!?；;:：、\-\(\)（）\"'“”]", t))
    digits = len(re.findall(r"\d", t))
    tokens = re.findall(r"\w+", t.lower())
    ttr = (len(set(tokens)) / max(1, len(tokens))) if tokens else 0.0
    avg_sent_len = n_words / max(1, n_sents)
    punct_ratio = punct / max(1, n_chars)
    digit_ratio = digits / max(1, n_chars)

    return np.array(
        [n_chars, n_words, n_sents, ttr, avg_sent_len, punct_ratio, digit_ratio],
        dtype=float
    ).reshape(1, -1)

def stats(text: str):
    t = text.strip()
    n_chars = len(t)
    words = re.findall(r"\w+", t)
    n_words = len(words)
    n_sents = max(1, len(re.findall(r"[。！？!?\.]+", t)))
    punct = len(re.findall(r"[，,。！？!?；;:：、\-\(\)（）\"'“”]", t))
    ttr = (len(set([w.lower() for w in words])) / max(1, len(words))) if words else 0.0
    return {
        "字元數": n_chars,
        "字詞數": n_words,
        "句子數": n_sents,
        "標點數": punct,
        "TTR(多樣性)": round(ttr, 3),
    }

st.title("AI / Human 文章偵測器（AI Detector）")
st.caption("Baseline：TF-IDF + 手工特徵 + Logistic Regression")

text = st.text_area("貼上要判斷的文字", height=220, placeholder="在這裡輸入一段文字…")

col1, col2 = st.columns([1, 1])
with col1:
    threshold = st.slider("判定閾值（AI>=）", 0.1, 0.9, 0.5, 0.05)
with col2:
    show_stats = st.checkbox("顯示統計量", value=True)

if st.button("立即分析", type="primary", disabled=(not text.strip())):
    vectorizer, model = load_artifacts()

    X_tfidf = vectorizer.transform([text])
    X_hand = basic_features(text)
    X = hstack([X_tfidf, X_hand])

    p_ai = float(model.predict_proba(X)[0, 1])
    p_h = 1.0 - p_ai

    st.subheader("判斷結果")
    st.metric("AI%", f"{p_ai*100:.1f}%")
    st.metric("Human%", f"{p_h*100:.1f}%")

    st.progress(min(max(p_ai, 0.0), 1.0))

    verdict = "偏 AI" if p_ai >= threshold else "偏 Human"
    st.success(f"結論：{verdict}（閾值 {threshold:.2f}）")

    if show_stats:
        st.subheader("統計量（可選加分）")
        s = stats(text)
        st.write(s)
