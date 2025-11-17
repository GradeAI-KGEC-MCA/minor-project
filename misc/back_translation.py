from transformers import MarianMTModel, MarianTokenizer
import torch
import json
import random
from tqdm import tqdm  

# Utils
def save_json(data, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4)

def get_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def combine_data(*args):
    combined = []
    for lst in args:
        combined.extend(lst)
    random.shuffle(combined)
    return combined

# Load original data
original_data = get_json('./data/augmented/synonym_replacement.json')

# Models
en_to_es_model_name = "Helsinki-NLP/opus-mt-en-es"
es_to_en_model_name = "Helsinki-NLP/opus-mt-es-en"

en_to_es_tokenizer = MarianTokenizer.from_pretrained(en_to_es_model_name)
en_to_es_model = MarianMTModel.from_pretrained(en_to_es_model_name).eval()
es_to_en_tokenizer = MarianTokenizer.from_pretrained(es_to_en_model_name)
es_to_en_model = MarianMTModel.from_pretrained(es_to_en_model_name).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
en_to_es_model.to(device)
es_to_en_model.to(device)

# Batched translation with progress
def batch_translate(texts, tokenizer, model, batch_size=16):
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs, max_new_tokens=512)
        translations.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    return translations

# Back-translate
answers = [r['provided_answer'] for r in original_data]
spanish_texts = batch_translate(answers, en_to_es_tokenizer, en_to_es_model)
back_translated_texts = batch_translate(spanish_texts, es_to_en_tokenizer, es_to_en_model)

back_translated_records = [
    {
        "question": r["question"],
        "reference_answer": r["reference_answer"],
        "provided_answer": bt,
        "verification_feedback": "incorrect",
        "score": 0,
        "max_score": r["max_score"]
    }
    for r, bt in zip(original_data, back_translated_texts)
]

# Combine and save

print("Original:", len(original_data))
print("Back-translated:", len(back_translated_records))
