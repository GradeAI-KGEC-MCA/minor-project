from transformers import AutoModelForSequenceClassification
import torch
from model.tokenizer import tokenize

def inference(data: dict)->None:
    loader = torch.utils.data.DataLoader(data['unseen_q'], batch_size=8)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(**{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]})
            preds = outputs.logits.squeeze().tolist()
            predictions.extend(preds)
    

    print(predictions[:10])

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
inference(data=tokenized_data)