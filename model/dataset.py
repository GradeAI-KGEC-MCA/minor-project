import torch
from torch.utils.data import Dataset
from model.weight import LABEL2ID, compute_sample_weight

class QAClassifierDataset(Dataset):
    
    def __init__(self, samples, tokenizer, class_weight, max_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.class_weight = class_weight
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        text = (
            f"Question: {s['question']}\n"
            f"Reference Answer: {s['reference_answer']}\n"
            f"Student Answer: {s['provided_answer']}"
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        label = LABEL2ID[s["verification_feedback"]]
        weight = compute_sample_weight(s, self.class_weight)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label),
            "weights": torch.tensor(weight, dtype=torch.float)
        }
