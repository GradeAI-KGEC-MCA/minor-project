import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load dataset
with open('./data/training_questions.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)

# Load model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")


def gen_incorrect_answer(question, reference, max_score, temperature=0.9, top_p=0.9, max_tokens=200):
    prompt = f"""
You are generating synthetic student answers for training a grading model.

Question: {question}
Reference Answer: {reference}
Maximum Marks: {max_score}

Generate one student answer that:
- Appears related to the question
- Is factually or conceptually incorrect compared to the reference
- Sounds human-written and coherent
- Does not copy or paraphrase the reference answer

Answer:
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
    return text.split("Answer:")[-1].strip()

def augment_incorrect(dataset, per_question=5, output_path="./data/augmented_incorrect.json"):
    augmented = []
    for i, record in enumerate(dataset):
        q = record["question"]
        ref = record["reference_answer"]
        mx = record["max_score"]

        for _ in range(per_question):
            incorrect = gen_incorrect_answer(q, ref, mx)
            augmented.append({
                "question": q,
                "reference_answer": ref,
                "student_answer": incorrect,
                "verification_feedback": "incorrect",
                "score": 0,
                "max_score": mx
            })

        if i % 5 == 0:
            print(f"Processed {i}/{len(dataset)} questions")

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(augmented, f, indent=4)
    print(f"Saved {len(augmented)} synthetic samples to {output_path}")
    return augmented

# Run generation
augmented_data = augment_incorrect(dataset, per_question=5)
