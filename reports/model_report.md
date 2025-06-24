# 🧠 Model Report: Amazon Review Sentiment Classifier

## ✅ Model Architecture Decisions
- **Base Model**: `distilbert-base-uncased` (lightweight, efficient BERT variant)
- **Tokenizer**: Hugging Face `AutoTokenizer` with WordPiece vocabulary
- **Max Sequence Length**: 256 tokens (covers most reviews)
- **Head**: Pre-classification linear layer with softmax for binary output

---

## 📈 Training Insights
- **Framework**: Hugging Face `Trainer` API
- **Dataset**: `amazon_polarity` (balanced binary classification)
- **Batch Size**: 16
- **Epochs**: 3
- **Optimizer**: AdamW (default)
- **Evaluation Strategy**: per epoch
- **Metrics**: Accuracy, F1, Precision, Recall

---

## 🔧 Observations & Improvements
- Model achieved **~94–95%** performance across all metrics.
- Balanced dataset led to stable and reliable evaluation results.
- No signs of overfitting after 3 epochs.
- Could benefit from:
  - **Learning rate tuning**
  - **Cross-validation**
  - **Multilingual extension** (e.g., using `xlm-roberta-base`)

