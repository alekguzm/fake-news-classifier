[README (1).md](https://github.com/user-attachments/files/26326695/README.1.md)
# Fake News Classifier

A lightweight machine learning pipeline that classifies news articles as **FAKE** or **REAL** using TF-IDF features and a Stochastic Gradient Descent (SGD) classifier.

---

## Overview

This project trains a text classification model on labeled news data. It vectorizes article text using TF-IDF and trains an SGD classifier with a modified Huber loss function, which provides probability estimates alongside predictions.

---

## Project Structure

```
.
├── classifier.py   # Main script: load, train, evaluate
├── news.csv        # Dataset (text, label columns)
└── README.md
```

---

## Dataset Format

The model expects a CSV file (`news.csv`) with at least two columns:

| Column  | Description                        |
|---------|------------------------------------|
| `text`  | The full article body              |
| `label` | Class label — either `FAKE` or `REAL` |

Malformed rows are automatically skipped during loading.

---

## Configuration

All key hyperparameters are defined as constants at the top of `classifier.py`:

| Constant       | Default | Description                              |
|----------------|---------|------------------------------------------|
| `TEST_SIZE`    | `0.2`   | Fraction of data used for testing        |
| `RANDOM_STATE` | `7`     | Seed for reproducibility                 |
| `MAX_ITER`     | `50`    | Maximum training iterations for SGD      |
| `MAX_DF`       | `0.7`   | Ignores terms appearing in >70% of docs  |

---

## Usage

### 1. Install dependencies

```bash
pip install pandas scikit-learn
```

### 2. Run the classifier

```bash
python classifier.py
```

### 3. Expected output

```
Accuracy: 0.9284
Confusion Matrix:
[[589  48]
 [ 43 587]]
```

---

## How It Works

1. **Load** — Reads `news.csv`, extracting the `text` and `label` columns
2. **Split** — Divides data into 80% train / 20% test sets
3. **Vectorize** — Converts raw text to TF-IDF feature vectors, filtering English stopwords and high-frequency terms
4. **Train** — Fits an `SGDClassifier` using the modified Huber loss
5. **Evaluate** — Reports accuracy and a confusion matrix on the held-out test set

---

## Model Details

| Component    | Choice                        | Why                                                   |
|--------------|-------------------------------|-------------------------------------------------------|
| Vectorizer   | `TfidfVectorizer`             | Balances term frequency with document rarity          |
| Classifier   | `SGDClassifier`               | Fast, scalable, works well with sparse text features  |
| Loss         | `modified_huber`              | Robust to outliers; enables probability estimates     |
| Stop words   | English built-in              | Removes low-signal common words                       |

---

## Potential Improvements

- **Save the model** — Use `joblib.dump()` to persist the trained model and vectorizer for reuse without retraining
- **Null handling** — Add `dropna()` in `load_data()` to guard against missing values
- **Richer metrics** — Add `classification_report()` for per-class precision, recall, and F1
- **Inference function** — Expose a `predict(text)` helper for use outside the `__main__` block
- **Cross-validation** — Replace a single train/test split with k-fold CV for more reliable evaluation

---

## License

MIT
