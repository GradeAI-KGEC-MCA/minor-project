from datasets import load_dataset, Dataset
from transformers import BertTokenizerFast

_default_tokenizer = 'bert-base-uncased'

def _load_datasets(data_files: dict) -> Dataset:
    return load_dataset('json', data_files=data_files)

def tokenize(data_files: dict, tokenizer_path: str = _default_tokenizer, is_training: bool = True) -> dict:
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    data = _load_datasets(data_files)
    
    # All original columns to be removed after processing
    # We detect them dynamically from the first available split
    sample_split = list(data.keys())[0]
    columns_to_remove = data[sample_split].column_names
    
    def create_tokens(batch: dict) -> dict:
        # Format: Segment A = Question + Reference, Segment B = Student Answer
        combined_context = [
            f"Question: {q} Reference: {r}" 
            for q, r in zip(batch["question"], batch["reference_answer"])
        ]
        
        encodings = tokenizer(
            combined_context,           # Sequence A
            batch["provided_answer"],   # Sequence B
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        if is_training and "verification_feedback" in batch:
            # Map "correct" -> 1, others -> 0
            encodings["labels"] = [1 if v == "correct" else 0 for v in batch["verification_feedback"]]
        
        return encodings

    # Apply tokenization and clean up columns
    data = data.map(create_tokens, batched=True, batch_size=8, remove_columns=columns_to_remove)
    data.set_format(type="torch")
    
    if tokenizer_path == _default_tokenizer:
        tokenizer.save_pretrained("./model/bert_tokenizer")
    
    return data