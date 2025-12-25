# AI / Human 文章偵測器（AI Detector）

本專案實作一個簡易的 AI vs Human 文章分類系統，使用傳統機器學習方法建立 baseline，並透過 Streamlit 提供即時互動介面。

## 方法

- 特徵擷取：
  - TF-IDF（1–2 grams）
  - 手工特徵（字數、句子數、標點比例、詞彙多樣性）
- 分類模型：Logistic Regression
- 任務：Binary Classification（AI / Human）

## 專案結構

## 使用方式

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py

---
