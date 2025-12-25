# AI / Human 文章偵測器（AI Detector）

本專案為課程作業之實作，目標為建立一個簡易的 AI vs Human 文章分類工具。
系統可即時輸入文字，並輸出該文本為 AI 生成或人類撰寫的機率判斷結果。

---

## 一、方法說明

本作業採用傳統機器學習方式建立 baseline 模型，流程如下：

- 特徵擷取：
  - TF-IDF（1–2 gram）
  - 手工特徵（字數、句子數、標點比例、詞彙多樣性等）
- 分類模型：
  - Logistic Regression
- 任務類型：
  - Binary Classification（AI / Human）

---

## 二、系統架構

- `train.py`：模型訓練程式（離線訓練）
- `app.py`：即時推論與使用者介面（Streamlit）
- `data/`：訓練資料
- `artifacts/`：訓練完成後輸出的模型與向量器

---

## 三、使用方式

### 1️⃣ 安裝套件
```bash
pip install -r requirements.txt
