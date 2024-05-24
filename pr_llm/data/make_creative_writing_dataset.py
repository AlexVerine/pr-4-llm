from argparse import ArgumentParser

import numpy as np
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from tqdm import tqdm

from pr_llm.utils import get_env

writing_prompts = [
    "Write about a dream you had.",
    "Create and write about a new character.",
    "Write about a place you'd love to visit.",
    "Write about an important life event.",
    "Write about life 100 years from now.",
    "Write a story where magic exists in everyday life.",
    "Write a poem about a personal experience.",
    "Write a speech for a cause you believe in.",
    "Write a short mystery story.",
    "Write a modern day fairy tale.",
    "Write a story set in a historical period.",
    "Write a story about a technological advancement.",
    "Write a letter to your future self.",
    "Write a story from an animal's perspective.",
    "Write a week's worth of diary entries for a character.",
    "Write a short story using mythological characters.",
    "Write a conversation between two characters.",
    "Write about a day in your life.",
    "Write a series of Haikus about seasons.",
    "Describe a place without naming it.",
    "Write a news article about an event in your town.",
    "Write a recipe with a story.",
    "Write a one-act play.",
    "Write a story set in a dystopian future.",
    "Write a satirical essay on a trending topic.",
    "Write a humorous comic strip.",
    "Write a piece inspired by nature.",
    "Write a short story about your favorite fictional character.",
    "Write instructions for an invented machine.",
    "Write a story using only metaphors.",
    "Write a script for a short film.",
    "Write a conversation in text message format.",
    "Write a limerick about a funny event.",
    "Write a story about a journey to space.",
    "Write a script for a documentary on a subject of your choice.",
    "Write a eulogy for a character from your favorite book.",
    "Write a script for a radio show.",
    "Write a song about a memorable event.",
    "Write a story with a hidden meaning.",
    "Write a fable with a moral.",
    "Write a comedic monologue.",
    "Write an opinion piece on a current event.",
    "Write a parody of a popular book or movie.",
    "Write a story where a character has a magical power.",
    "Write a story about time travel.",
    "Write a story based on an object in a mystery box.",
    "Write a ghost story.",
    "Describe a haunted house without using cliché descriptions.",
    "Write a story about a conspiracy theory.",
    "Write a story about a non-traditional superhero.",
]


def sample_examples(sampler, dataset):
    example_ids = sampler.integers(low=0, high=len(dataset), size=1)
    example = dataset[int(example_ids[0])]

    return example


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_samples", default=6000, type=int)
    parser.add_argument("--seed", default=12357, type=int)
    args = parser.parse_args()
    load_dotenv()
    DATA_PATH = get_env("DATA_PATH")
    rng = np.random.default_rng(seed=args.seed)
    examples = rng.choice(writing_prompts, args.max_samples, replace=True).tolist()

    dataset = Dataset.from_dict({"prompt": examples})
    dataset.save_to_disk(DATA_PATH / "creative_writings")
