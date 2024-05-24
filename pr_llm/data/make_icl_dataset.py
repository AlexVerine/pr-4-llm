from argparse import ArgumentParser

import numpy as np
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from tqdm import tqdm

from pr_llm.utils import get_env


def sample_examples(sampler, dataset, n_icl, max_example_words=250, max_title_words=10):
    examples_ids = sampler.integers(low=0, high=len(dataset), size=n_icl)
    examples = [dataset[int(i)] for i in examples_ids]
    clean_examples = []
    for ex in examples:
        title = ex["title"]
        title_words = title.split(" ")
        title = " ".join(title_words[:max_title_words])
        content = ex["content"]
        content_words = content.split(" ")
        content = " ".join(content_words[:max_example_words])
        clean_examples.append({"title": title, "content": content})

    return clean_examples


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--max_samples", default=100, type=int)
    parser.add_argument("--seed", default=12357, type=int)
    parser.add_argument("--n_icl", default=1, type=int)
    parser.add_argument("--max_example_words", default=350, type=int)
    parser.add_argument("--max_title_words", default=10, type=int)
    args = parser.parse_args()
    load_dotenv()
    DATA_PATH = get_env("DATA_PATH")

    dataset = load_from_disk(DATA_PATH / args.data_path)
    icl_dataset = {f"example_{n}": [] for n in range(args.n_icl)}
    rng = np.random.default_rng(seed=args.seed)
    for i in tqdm(range(args.max_samples)):
        examples = sample_examples(
            rng,
            dataset,
            args.n_icl,
            max_example_words=args.max_example_words,
            max_title_words=args.max_title_words,
        )
        for n, example in enumerate(examples):
            icl_dataset[f"example_{n}"].append(example)

    icl_dataset = Dataset.from_dict(icl_dataset)
    icl_dataset.save_to_disk(DATA_PATH / f"{args.data_path}_icl_{args.n_icl}")
