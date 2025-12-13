from misc.dataset_modifier import get_json
import pandas as pd

data = get_json('./data/augmented/aug.json')

# Convert to DataFrame
df = pd.DataFrame(data)

# Desired column order
desired_order = [
    "id",
    "question",
    "reference_answer",
    "provided_answer",
    "original_answer",
    "answer_feedback",
    "verification_feedback",
    "max_score",
    "normalized_score"
]

# Reorder only if columns exist
df = df[[col for col in desired_order if col in df.columns]]

# Export to Excel
df.to_excel("output.xlsx", index=False)
