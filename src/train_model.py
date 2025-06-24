from transformers import Trainer, TrainingArguments
from src.data_preprocessing import load_and_tokenize
from src.model_utils import get_model
import src.config as cfg
import os
import json

def train():
    dataset = load_and_tokenize(cfg.model_name, cfg.max_len)
    model = get_model(cfg.model_name)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds).tolist()
        metrics = {
            "accuracy": acc, "f1": f1, "precision": precision, "recall": recall,
            "confusion_matrix": cm
        }
        with open('./reports/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f)
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'].shuffle(seed=42).select(range(10000)),  # subsample
        eval_dataset=dataset['test'].select(range(2000)),  # subsample
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    model.config.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    train()
