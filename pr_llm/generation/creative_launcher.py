from argparse import ArgumentParser

from dotenv import load_dotenv
from omegaconf import OmegaConf

from pr_llm.generation.evaluation import run_generation_distributed

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to .yaml config file.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    load_dotenv()

    args = parser.parse_args()

    config = OmegaConf.load(f"configs/generation/creative/{args.config}.yaml")

    config.seed = args.seed
    print("Launching generation")
    run_generation_distributed(config)
