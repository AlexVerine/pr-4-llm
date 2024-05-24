from argparse import ArgumentParser

from datasets import load_from_disk
from dotenv import load_dotenv

from pr_llm.generation.generator import DistributedInferenceGenerator
from pr_llm.generation.utils import get_generative_model
from pr_llm.utils import get_env

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--writing_steps", default=200, type=int)
    parser.add_argument("--precision", default="16")
    parser.add_argument("--max_length", default=1, type=int)
    args = parser.parse_args()
    load_dotenv()
    DATASET_PATH = get_env("WIKIPEDIA_SENT_PATH")
    DATA_PATH = get_env("DATA_PATH")
    generative_model, tokenizer = get_generative_model(
        model=args.model,
        num_new_tokens=args.max_length,
        decoding_strategy="sample",
        precision=args.precision,
    )
    text_dataset = load_from_disk(DATASET_PATH)["train"]
    text_dataset = text_dataset
    output_path = DATA_PATH / f"{args.model}_{DATASET_PATH.stem}"
    inference_generator = DistributedInferenceGenerator(
        generative_model=generative_model,
        per_device_batch_size=args.batch_size,
        dataset=text_dataset,
        tokenize=True,
        output_file=output_path,
        writing_steps=args.writing_steps,
    )
    inference_generator.generate_text()
