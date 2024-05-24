from argparse import ArgumentParser

from pr_llm.evaluate.utils import get_generation_hparams, make_eval_parser


def make_creative_parser():
    base_parser = make_eval_parser()
    parser = ArgumentParser(parents=[base_parser], add_help=False)

    parser.add_argument("--config_a", type=str, required=True)
    parser.add_argument("--config_b", type=str, required=True)
    return parser


def get_hparams(config):
    hparams = {}
    hparams.update(get_generation_hparams(config))
    return hparams
