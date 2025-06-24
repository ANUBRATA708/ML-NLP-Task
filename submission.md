# 📝 Submission: ML/NLP Engineer Intern Challenge

## 🔍 Project Title
**Amazon Review Sentiment Classifier using Hugging Face Transformers**

---

## 🚀 My Approach

For this challenge, I built a complete text classification pipeline using the Hugging Face Transformers library. I selected the `amazon_polarity` dataset, which contains a large volume of labeled product reviews categorized into positive and negative sentiments — making it ideal for a binary classification task.

The process involved:

- Exploring the dataset using Jupyter notebooks to understand class distribution and review length.
- Preprocessing the text using Hugging Face’s `AutoTokenizer`.
- Fine-tuning the `distilbert-base-uncased` model using Hugging Face’s `Trainer` API.
- Evaluating the model using standard metrics like Accuracy, Precision, Recall, and F1 Score.
- Visualizing and analyzing errors with a confusion matrix.
- Saving the final model and tokenizer for later inference or deployment.

---

## 🧠 Model Decisions

- **Model**: I chose `distilbert-base-uncased` for its balance of performance and efficiency.
- **Tokenizer**: Used Hugging Face's `AutoTokenizer` to handle all encoding.
- **Sequence Length**: Set to 256 tokens to cover most reviews while maintaining speed.
- **Batch Size**: 16 (small enough for memory, large enough for stable gradients).
- **Epochs**: 3 (more epochs didn’t significantly improve performance).
- **Evaluation**: Performed at the end of each epoch with `Trainer`.

---

## 📊 Evaluation Summary

| Metric     | Score     |
|------------|-----------|
| Accuracy   | ~95%      |
| F1 Score   | ~94%      |
| Precision  | ~94%      |
| Recall     | ~94%      |

The model was evaluated on a held-out test set and the metrics were consistent, confirming that it generalizes well on unseen data.

---

## 🔧 What Went Well

- The dataset was already clean and well-balanced, which simplified preprocessing.
- Using Hugging Face’s Trainer significantly accelerated experimentation.
- DistilBERT proved to be fast and effective for sentiment classification.
- The modular folder structure made the project easy to scale and debug.

---

## 💡 Improvement Ideas

- Tune hyperparameters like learning rate or try training with `lr_scheduler`.
- Implement K-fold cross-validation for better performance estimation.
- Extend to a multilingual use case using models like `xlm-roberta-base`.
- Create a Streamlit or Gradio app to deploy the model with a user interface.

---

## 🎓 Key Learnings

- Hugging Face’s ecosystem is highly productive for NLP tasks — everything from tokenization to evaluation is well-integrated.
- Even small architecture changes (like max sequence length) can affect training time and accuracy.
- Visualizing performance with tools like confusion matrices helps pinpoint misclassifications.
- Structuring your ML project cleanly from the start pays off during debugging and submission.

---

Thanks for reviewing my submission! I really enjoyed building and learning from this project.
