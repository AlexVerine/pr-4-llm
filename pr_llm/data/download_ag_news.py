from datasets import load_dataset
from dotenv import load_dotenv

from pr_llm.utils import get_env

if __name__ == "__main__":
    load_dotenv()
    DATA_PATH = get_env("DATA_PATH")

    # Load from HuggingFace data hub. First argument is the dataset name.
    # Second is the subset.
    dataset = load_dataset("ag_news", split="train")

    dataset.save_to_disk(DATA_PATH / "ag_news")
