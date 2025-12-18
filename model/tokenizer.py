from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast

_default_tokenizer = 'bert-base-uncased'

def tokenize(data_dict: dict, tokenizer_path: str = _default_tokenizer, is_training: bool = True) -> DatasetDict:
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    hf_datasets = {split: Dataset.from_list(data_list) for split, data_list in data_dict.items()}
    data = DatasetDict(hf_datasets)
    
    # We NO LONGER define columns_to_remove. 
    # This keeps every field from every JSON split.

    def create_tokens(batch: dict) -> dict:
        combined_context = [f"Question: {q} Reference: {r}" for q, r in zip(batch["question"], batch["reference_answer"])]
        encodings = tokenizer(
            combined_context, 
            batch["provided_answer"], 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )
        if is_training and "verification_feedback" in batch:
            encodings["labels"] = [1 if v == "correct" else 0 for v in batch["verification_feedback"]]
        return encodings

    # Map without remove_columns
    data = data.map(create_tokens, batched=True, batch_size=8)
    
    # Crucial: set_format tells the model to ignore the 'stuff' and 'extra fields'
    # so they don't go into BERT as inputs.
    data.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "token_type_ids"], 
        output_all_columns=True 
    )
    return data