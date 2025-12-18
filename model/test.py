import torch
import json
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F
from model.tokenizer import tokenize
from misc.dataset_modifier import get_json, separate, combine_data, find_data, save_json

def inference(data, model, original_data: list, output_file_path: str):
    # Use the DataLoader for the model inputs only
    loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False)
    
    predictions = []
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        print('Starting inference...')
        for i, batch in enumerate(loader):
            # Model only receives the necessary keys
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}            
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).tolist()
            preds = ["correct" if p == 1 else "incorrect" for p in preds]
            predictions.extend(preds)
            
            if i % 10 == 0:
                print(f"Batch {i} processed...")

    # Reconstruct output using the original 'id' from the dataset object
    output = []
    for i in range(len(predictions)):
        record = find_data(original_data, data[i]['id'])
        print(data[i]['id'])
        output.append({
            'id': data[i]['id'],
            'question': record.get('question'),
            'original_feedback': record.get('verification_feedback'),
            'model_feedback': predictions[i]
        })

    save_json(output, output_file_path)
    print(f"Saved {len(output)} predictions to {output_file_path}")


# Paths
original_file_q = get_json("./data/updated/formatted/unseen_questions.json")
original_file_a = get_json("./data/curated/unseen_answers.json")
output_file_q = "./model/test_results/unseen_q.json"
output_file_a = "./model/test_results/unseen_a.json"


# Tokenize unseen data (is_training=False)
data_files = {
    "unseen_q": original_file_q,
    "unseen_a": original_file_a
}
q = separate(original_file_q)
a = separate(original_file_a)
data_files = {
    "unseen_q": combine_data(q['correct']+q['incorrect']),
    "unseen_a": combine_data(a['correct']+a['incorrect'])
}

tokenized_data = tokenize(data_dict=data_files, tokenizer_path='./model/bert_tokenizer', is_training=False)

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained('./model/results')
model.eval()

# Run inference
inference(tokenized_data['unseen_q'], model, data_files['unseen_q'], output_file_q)
inference(tokenized_data['unseen_a'], model, data_files['unseen_a'], output_file_a)
