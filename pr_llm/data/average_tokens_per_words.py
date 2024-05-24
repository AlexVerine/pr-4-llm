import argparse

import numpy as np
from datasets import load_from_disk
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from pr_llm.utils import get_env
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", type=str, help="path to the model")
    args = parser.parse_args()
    load_dotenv()
    checkpoint_path = get_env("CHECKPOINT_PATH")

    data_path = get_env("DATA_PATH")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path / args.model))

    dataset = load_from_disk(data_path / "webtext" / "train")

    def count_tokens_and_words(example):
        tokens = len(tokenizer.encode(example["text"], add_special_tokens=False))
        words = len(word_tokenize(example["text"]))
        return {"tokens": tokens, "words": words}

    dataset = dataset.map(count_tokens_and_words, num_proc=16)

    tokens = np.array(dataset["tokens"])
    words = np.array(dataset["words"])
    print("Tokens per word:", np.mean(tokens.sum() / words.sum()))
