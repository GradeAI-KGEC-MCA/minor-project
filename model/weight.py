from collections import Counter
import math
from misc.dataset_modifier import get_json

LABEL2ID = {
    "incorrect": 0,
    "partial": 1,
    "correct": 2
}

def compute_class_weights(samples):
    counts = Counter(s["verification_feedback"] for s in samples)
    total = sum(counts.values())

    class_weight = {
        lbl: math.sqrt(total / cnt)
        for lbl, cnt in counts.items()
    }

    # normalize to mean = 1.0
    mean_w = sum(class_weight.values()) / len(class_weight)
    for k in class_weight:
        class_weight[k] /= mean_w

    return class_weight

def compute_sample_weight(sample, class_weight):
    # class weight
    w = class_weight[sample["verification_feedback"]]

    # source weight
    if sample.get("is_augmented", False):
        conf = sample["confidence"]  # guaranteed >= 0.5
        # map confidence [0.5, 1.0] → weight [0.5, 0.7]
        w_source = 0.5 + (conf - 0.5) * (0.7 - 0.5) / 0.5
    else:
        w_source = 1.0

    w *= w_source

    # clamp
    w = max(0.1, min(w, 2.0))
    return w

data = get_json('./data/updated/combined_set/train.json')

class_weights = compute_class_weights(data)

print(class_weights)