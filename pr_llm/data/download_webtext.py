import json
import os

import numpy as np
import requests
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from tqdm import tqdm

from pr_llm.utils import get_env


def load_json_dataset(data_dir, dataset_name, split=None, max_num_data=np.inf):
    if split is None:
        path = os.path.join(data_dir, f"{dataset_name}.jsonl")
    else:
        path = os.path.join(data_dir, f"{dataset_name}.{split}.jsonl")
    texts = []
    for i, line in enumerate(open(path)):
        if i >= max_num_data:
            break
        texts.append(json.loads(line)["text"])
    return texts


if __name__ == "__main__":
    subdir = "data"
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace("\\", "/")  # needed for Windows

    for ds in ["webtext"]:
        for split in ["train", "valid", "test"]:
            filename = ds + "." + split + ".jsonl"
            r = requests.get(
                "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/"
                + filename,
                stream=True,
            )

            with open(os.path.join(subdir, filename), "wb") as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(
                    ncols=100,
                    desc="Fetching " + filename,
                    total=file_size,
                    unit_scale=True,
                ) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)

    def load_json_dataset(data_dir, dataset_name, split=None, max_num_data=np.inf):
        if split is None:
            path = os.path.join(data_dir, f"{dataset_name}.jsonl")
        else:
            path = os.path.join(data_dir, f"{dataset_name}.{split}.jsonl")
        texts = []
        for i, line in enumerate(open(path)):
            if i >= max_num_data:
                break
            texts.append(json.loads(line)["text"])
        return texts

    dataset = DatasetDict()
    for split in ["train", "valid", "test"]:
        dataset[split] = Dataset.from_dict(
            {
                "text": load_json_dataset(
                    data_dir="data", dataset_name="webtext", split=split
                )
            }
        ).shuffle(seed=12345)
    load_dotenv()
    k_words = 10
    DATA_PATH = get_env("DATA_PATH")
    dataset.save_to_disk(DATA_PATH / "webtext")
    webtext_10k = dataset["train"].filter(lambda x: len(x["text"].split(" ")) > k_words)
    webtext_10k = webtext_10k.shuffle(seed=12345).select(range(15000))
    webtext_10k.save_to_disk(DATA_PATH / f"webtext_15k_{k_words}w")
