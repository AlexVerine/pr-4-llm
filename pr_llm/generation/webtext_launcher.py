from argparse import ArgumentParser

from dotenv import load_dotenv
from omegaconf import OmegaConf

from pr_llm.generation.evaluation import run_generation_distributed


def launch_webtext(config):
    load_dotenv()

    print("Launching generation")
    run_generation_distributed(config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to .yaml config file.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()
    config = OmegaConf.load(f"configs/generation/webtext/{args.config}.yaml")
    config.seed = args.seed

    launch_webtext(config)
