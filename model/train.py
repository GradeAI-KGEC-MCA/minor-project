from model.tokenizer import tokenize
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

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
    weight_decay=0.01
)
        
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"]
)

trainer.train()
model.save_pretrained("./model/bert_grader")