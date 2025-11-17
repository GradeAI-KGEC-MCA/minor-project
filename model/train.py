# train.py
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from model.tokenizer import tokenize

# Safe metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "f1": f1}

# Custom Trainer with CrossEntropyLoss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Load datasets
data_files = {
    "train": "./data/updated/combined/train.json",
    "validation": "./data/updated/combined/train.json"
}
tokenized_data = tokenize(data_files=data_files, is_training=True)

# Check label distribution
train_labels = [x["labels"].item() for x in tokenized_data["train"]]
print("Train labels distribution:", {0: train_labels.count(0), 1: train_labels.count(1)})

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model/results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=8,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    load_best_model_at_end=True,
    weight_decay=0.05
)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    compute_metrics=compute_metrics
)

# Train and save
trainer.train()
model.save_pretrained("./model/bert_safe")
