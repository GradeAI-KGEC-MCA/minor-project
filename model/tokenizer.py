from datasets import load_dataset, Dataset
from re import sub
from transformers import BertTokenizerFast

_default_tokenizer = 'bert-base-uncased'

def _clean_data(text: str) -> str:
    return sub(r"\s+", " ", text.strip())

def _preprocess(example: dict) -> dict:
    reference = _clean_data(example["reference_answer"])
    student = _clean_data(example["provided_answer"])
    example["input_text"] = f"Reference: {reference} Student: {student}"
    return example

def _load_datasets(data_files: set) -> Dataset:
    return load_dataset('json', data_files=data_files)

def tokenize(data_files: set, tokenizer_path: str = _default_tokenizer, is_training: bool = True) -> dict:
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    data = _load_datasets(data_files)
    
    # Preprocess to add input_text
    data = data.map(_preprocess)
    
    # Columns to remove after tokenization
    columns_to_remove = [
        "question",
        "reference_answer",
        "provided_answer",
        "answer_feedback",
        "verification_feedback",
        "normalized_score",
        "max_score"
    ]
    
    # Tokenization function
    def create_tokens(batch: dict) -> dict:
        encodings = tokenizer(
            batch["input_text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        if is_training and "verification_feedback" in batch:
            # Map "correct"/"incorrect" to 1/0
            encodings["labels"] = [1 if v == "correct" else 0 for v in batch["verification_feedback"]]
        return encodings

    data = data.map(create_tokens, batched=True, batch_size=8)
    data = data.remove_columns(columns_to_remove)
    data.set_format(type="torch")
    
    if tokenizer_path == _default_tokenizer:
        tokenizer.save_pretrained("./model/bert_tokenizer(0)")
    
    return data
