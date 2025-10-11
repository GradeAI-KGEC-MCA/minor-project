import os
import json
import pandas as pd
import kagglehub
from datasets import load_dataset

def save_hf_dataset(dataset_name: str, out_prefix: str = "hf_dataset"):
    """
    Download a Hugging Face dataset and save each split as both Excel and JSON.
    """
    ds = load_dataset(dataset_name)
    print(f"Loaded splits: {list(ds.keys())}")

    for split in ds.keys():
        df = ds[split].to_pandas()

        excel_path = f"{out_prefix}_{split}.xlsx"
        df.to_excel(excel_path, index=False)

        json_path = f"{out_prefix}_{split}.json"
        df.to_json(json_path, orient="records", force_ascii=False)

        print(f"Saved {split} → {excel_path}, {json_path}")


def save_kaggle_dataset_json(dataset_name: str, out_prefix: str = "kaggle_dataset"):
    path = kagglehub.dataset_download(dataset_name)
    print("Downloaded Kaggle dataset at:", path)

    files = os.listdir(path)
    print("Files inside:", files)

    for fname in files:
        fpath = os.path.join(path, fname)
        name, ext = os.path.splitext(fname)

        if ext.lower() == ".csv":
            df = pd.read_csv(fpath)
        elif ext.lower() == ".json":
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.json_normalize(data)
        else:
            print(f"Skipping unsupported file: {fname}")
            continue

        excel_path = f"{out_prefix}_{name}.xlsx"
        df.to_excel(excel_path, index=False)

        json_path = f"{out_prefix}_{name}.json"
        df.to_json(json_path, orient="records", force_ascii=False)

        print(f"Saved {fname} → {excel_path}, {json_path}")

def save_kaggle_dataset_csv(dataset_name: str, out_prefix: str = ""):
    path = kagglehub.dataset_download(dataset_name)
    print("Downloaded Kaggle dataset at:", path)
    csv_to_json_xl(path, out_prefix)

def csv_to_json_xl(path: str, out_prefix: str = ""):
    files = os.listdir(path)
    print("Files inside:", files)

    for fname in files:
        fpath = os.path.join(path, fname)
        name, ext = os.path.splitext(fname)
        print(ext)

        if ext.lower() == ".csv":
            df = pd.read_csv(fpath)
        else:
            print(f"Skipping unsupported file: {fname}")
            continue

        excel_path = f"{out_prefix}_{name}.xlsx"
        print("0")
        df.to_excel(excel_path, index=False)

        json_path = f"{out_prefix}_{name}.json"
        print("1")
        df.to_json(json_path, orient="records", force_ascii=False)
        print("2")

        print(f"Saved {fname} → {excel_path}, {json_path}")

csv_to_json_xl(path="/home/mav204/Documents/programs/python/NHANES_2017-2018")