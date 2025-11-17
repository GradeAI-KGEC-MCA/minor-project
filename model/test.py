from transformers import AutoModelForSequenceClassification
import torch
from model.tokenizer import tokenize
import json

def inference(data, output_file_name:str, original_file_path: str)->None:
    loader = torch.utils.data.DataLoader(data, batch_size=8)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(**{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]})
            preds = outputs.logits.squeeze().tolist()
            if isinstance(preds, float):
                preds = [preds]
            predictions.extend(preds)
    
    predictions = [round(float(p), 1) for p in predictions]

    with open(original_file_path, 'r', encoding='utf8') as f:
        original_file = json.load(f)
    
    output = []
    for i, entry in enumerate(original_file):
        output.append({
            'id': entry['id'],
            'original_score': entry['score'],
            'model_score': predictions[i] if i < len(predictions) else None
        })
    
    with open('./model/test_results/' + output_file_name, 'w', encoding='utf8') as f:
        json.dump(output, f, indent=4)
    
    print(f'Saved {len(output)}predictions')


data_files={
    "unseen_q": "./data/unseen_questions.json",
    "unseen_a": "./data/unseen_answers.json"
}
tokenized_data = tokenize(data_files=data_files,
                          tokenizer_path='./model/bert_grader',
                          is_training=False
                          )

model = AutoModelForSequenceClassification.from_pretrained('./model/results')
model.eval()

print('Test Model loaded')
inference(
    data=tokenized_data['unseen_q'],
    output_file_name='unseen_q.json',
    original_file_path='./data/unseen_questions.json'
    )
inference(
    data=tokenized_data['unseen_q'],
    output_file_name='unseen_a.json',
    original_file_path='./data/unseen_answers.json'
    )