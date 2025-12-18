import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from model.tokenizer import tokenize

# --- METRICS ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    # Use macro F1 to give equal weight to both classes despite the skew
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}

# --- CUSTOM TRAINER WITH WEIGHTED LOSS ---
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # CrossEntropy with weights for handling skew
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# --- DATA PREPARATION ---
data_files = {"train": "./data/updated/combined/train.json"}
# Note: tokenized_data["train"] contains the full dataset
dataset_dict = tokenize(data_files=data_files, is_training=True)
full_dataset = dataset_dict["train"]

# Extract labels for stratification and weight calculation
all_labels = [x["labels"].item() for x in full_dataset]
counts = np.bincount(all_labels)
total = len(all_labels)

# Calculate weights: Weight = Total / (Num_Classes * Count_per_Class)
class_weights = torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float)
print(f"Label distribution: {counts}")
print(f"Applying Class Weights: {class_weights}")

# Stratified Split (90% Train, 10% Val)
indices = np.arange(len(full_dataset))
train_idx, val_idx = train_test_split(
    indices, 
    test_size=0.1, 
    stratify=all_labels, 
    random_state=42
)

train_dataset = full_dataset.select(train_idx)
eval_dataset = full_dataset.select(val_idx)

# --- MODEL & TRAINING ---
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./model/results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=8,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro", # Best model based on F1, not accuracy
    weight_decay=0.05,
    report_to="none"
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

# Start Training
trainer.train()

# Save final model
model.save_pretrained("./model/bert_safe")