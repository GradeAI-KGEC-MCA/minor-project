from datasets import load_dataset, Dataset
from re import sub
from transformers import BertTokenizerFast

_default_tokenizer = 'bert-base-uncased'

def _clean_data(data: str) -> str:
    data = sub(r"\s+", " ", data.strip())
    return data

def _preprocess(data: Dataset) -> Dataset:
    question = _clean_data(data["question"])
    reference = _clean_data(data["reference_answer"])
    student = _clean_data(data["provided_answer"])

    data["input_text"] = f"Question: {question} Reference: {reference} Student: {student}"
    return data

def _load_datasets(datafiles: set)->Dataset:
    data = load_dataset(
        'json',
        data_files=datafiles
    )
    return data

def tokenize(data_files:set, tokenizer_path:str = _default_tokenizer, is_training: bool = True) -> dict:
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    data = _load_datasets(datafiles=data_files)
    data = data.map(_preprocess)
    feature_columns_to_be_removed = ["question",
                                    "reference_answer",
                                    "provided_answer",
                                    "input_text",
                                    "answer_feedback",
                                    "verification_feedback", 
                                    "id",
                                    "score"
                                    ]

    def create_tokens(data:dict) -> dict:
        encoding = tokenizer(
            data["input_text"],
            padding="max_length",
            truncation=True,
            max_length=512 # The maximum number of tokens allowed by BERT
        )
        if is_training and "score" in data:
            encoding["labels"] = data["score"]

        return encoding
    
    data = data.map(create_tokens, batched=True, batch_size=8)
    data = data.remove_columns(feature_columns_to_be_removed)
    data.set_format("torch")

    if tokenizer_path == _default_tokenizer:
        tokenizer.save_pretrained("./bert_grader")
    return data

