from argparse import ArgumentParser

from dotenv import load_dotenv
from omegaconf import OmegaConf

from pr_llm.generation.evaluation import run_generation_distributed


def launch_wikipedia_bio(config):
    load_dotenv()

    print("Launching generation")
    run_generation_distributed(config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to .yaml config file.")
    parser.add_argument("--n_icl", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    args = parser.parse_args()
    load_dotenv()

    config = OmegaConf.load(f"configs/generation/wikipedia_bio/{args.config}.yaml")

    config.model.task.n_icl = args.n_icl
    config.seed = args.seed
    print("Launching generation")
    run_generation_distributed(config)
