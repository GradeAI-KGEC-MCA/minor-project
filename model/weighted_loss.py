import torch.nn as nn

class WeightedTrainerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels, weights):
        loss = self.ce(logits, labels)
        return (loss * weights).mean()
