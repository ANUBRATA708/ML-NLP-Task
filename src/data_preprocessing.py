from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize(tokenizer_name='distilbert-base-uncased', max_len=256):
    dataset = load_dataset("amazon_polarity")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_fn(examples):
        return tokenizer(examples["content"], padding="max_length", truncation=True, max_length=max_len)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["title", "content"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    return tokenized_dataset
