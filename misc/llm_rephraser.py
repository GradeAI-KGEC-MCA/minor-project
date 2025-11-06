import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load dataset
with open('./data/augmented_train.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)

# Load model
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def gen_incorrect_answer(answer, temperature=0.9, top_p=0.9, max_tokens=200):
    prompt = f"""
Rephrase the following answer to make it look different in wording and structure but keep its meaning identical.
Do not fix or improve reasoning or factual errors. Keep the same level of incorrectness.
Return only the rephrased answer.

Answer: "{answer}"
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    return text.strip().strip('"')

def augment_incorrect(dataset, output_path="./data/augmented_incorrect_rephrase.json"):
    augmented = []
    for i, record in enumerate(dataset):
        incorrect = gen_incorrect_answer(record['provided_answer'])
        augmented.append({
            "question": record['question'],
            "reference_answer": record['reference_answer'],
            "provided_answer": incorrect,
            "answer_feedback": "",
            "verification_feedback": "incorrect",
            "score": 0,
            "max_score": record['max_score']

        })
        if i % 10 == 0:
            print(f"Processed {i}/{len(dataset)}")

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(augmented, f, indent=4)
    print(f"Saved {len(augmented)} augmented samples to {output_path}")

# Run once per record
augment_incorrect(dataset)
