from argparse import ArgumentParser

from pr_llm.evaluate.utils import get_generation_hparams, make_eval_parser


def make_wikipedia_bio_parser():
    base_parser = make_eval_parser()
    parser = ArgumentParser(parents=[base_parser], add_help=False)

    parser.add_argument(
        "--n_icl",
        nargs="+",
        default=list(range(1, 13)),
        type=int,
        help="param for n_icl",
    )
    return parser


def get_hparams(config):
    hparams = {}
    hparams.update(get_generation_hparams(config))
    return hparams
