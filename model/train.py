from model.tokenizer import tokenize
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch import nn

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.squeeze(preds)
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    return {"mse": mse, "mae": mae, "r2": r2}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        loss_fct = nn.HuberLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

data_files = {
    "train": "./data/train.json",
    "validation": "./data/validation.json"
}

tokenized_data = tokenize(data_files=data_files)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1
)

training_args = TrainingArguments(
    output_dir="./model/results",
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    num_train_epochs=12,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    load_best_model_at_end=True,
    weight_decay=0.05
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("./model/bert_tokenizer")
