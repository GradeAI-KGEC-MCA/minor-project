from transformers import Trainer
from model.weighted_loss import WeightedTrainerLoss

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights")

        outputs = model(**inputs)
        logits = outputs.logits

        if self.state.global_step % 50 == 0:  # log every 50 steps
            print("Step:", self.state.global_step)
            print("Logits:", logits[:5])
            print("Labels:", labels[:5])
            print("Weights:", weights[:5])
            
        loss_fn = WeightedTrainerLoss()
        loss = loss_fn(logits, labels, weights)

        return (loss, outputs) if return_outputs else loss
