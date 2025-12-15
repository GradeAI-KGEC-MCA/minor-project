import torch
from torch.utils.data import Dataset
from model.weight import LABEL2ID, compute_sample_weight

class QAClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, class_weights, is_train=True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.class_weights = class_weights
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]

        input_text = (
            f"Question:\n{s['question']}\n\n"
            f"Reference Answer:\n{s['reference_answer']}\n\n"
            f"Student Answer:\n{s['provided_answer']}"
        )

        enc = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        print(item)
        item["labels"] = LABEL2ID[s["verification_feedback"]]

        if self.is_train:
            item["weights"] = compute_sample_weight(s, self.class_weights)
        else:
            item["weights"] = torch.tensor(1.0)

        return item
