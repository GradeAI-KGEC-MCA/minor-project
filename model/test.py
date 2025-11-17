import torch
import json
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F
from model.tokenizer import tokenize

def inference(data, model, original_file_path: str, output_file_path: str):
    loader = torch.utils.data.DataLoader(data, batch_size=16)
    predictions = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            logits = model(**inputs).logits
            preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1).tolist()
            # Map 1 → "correct", 0 → "incorrect"
            preds = ["correct" if p == 1 else "incorrect" for p in preds]
            predictions.extend(preds)

    # Load original file
    with open(original_file_path, 'r', encoding='utf8') as f:
        original_data = json.load(f)

    output = []
    for i, entry in enumerate(original_data):
        output.append({
            "question": entry.get("question"),
            "original_feedback": entry.get("verification_feedback", None),
            "model_feedback": predictions[i] if i < len(predictions) else None
        })

    # Save predictions
    with open(output_file_path, 'w', encoding='utf8') as f:
        json.dump(output, f, indent=4)

    print(f"Saved {len(output)} predictions to {output_file_path}")


# Paths
original_file_q = "./data/updated/combined/unseen_questions.json"
original_file_a = "./data/updated/combined/unseen_answers.json"
output_file_q = "./model/test_results/unseen_q.json"
output_file_a = "./model/test_results/unseen_a.json"

# Tokenize unseen data (is_training=False)
data_files = {
    "unseen_q": original_file_q,
    "unseen_a": original_file_a
}

tokenized_data = tokenize(data_files=data_files, tokenizer_path='./model/bert_tokenizer', is_training=False)

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained('./model/results')
model.eval()

# Run inference
inference(tokenized_data['unseen_q'], model, original_file_q, output_file_q)
inference(tokenized_data['unseen_a'], model, original_file_a, output_file_a)
