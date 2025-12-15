import torch
from torch.utils.data import Dataset
from model.weight import LABEL2ID, compute_sample_weight

class QAClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, class_weights, is_train=True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.class_weights = class_weights
        self.is_train = is_train

    def __getitem__(self, idx):
        s = self.samples[idx]

        enc = self.tokenizer(
            s["input_text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = LABEL2ID[s["verification_feedback"]]

        if self.is_train:
            item["weights"] = compute_sample_weight(s, self.class_weights)
        else:
            item["weights"] = torch.tensor(1.0)

        return item
