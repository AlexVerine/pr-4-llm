from argparse import ArgumentParser

import numpy as np
from omegaconf import OmegaConf


def make_same_length(p, q):
    if p.shape[0] == q.shape[0]:
        return p, q
    # Find the maximum length among the texts in both splits
    max_length = min(p.shape[0], q.shape[0])

    # Truncate or pad the texts in both splits to have the same length
    trunc_p = p[:max_length]
    trunc_q = q[:max_length]
    return trunc_p, trunc_q


def get_args_layer(layer):
    if layer == "-1":
        return -1
    elif layer == "0.5":
        return 0.5
    elif layer == "0":
        return 0


def make_eval_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Whether to recompute the features or load from the cache.",
    )
    parser.add_argument("--recompute_cpu", action="store_true")
    parser.add_argument("--recompute_gpu", action="store_true")
    parser.add_argument("--date", default="latest", type=str, help="param for date")
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument(
        "--max_samples", default=4000, type=int, help="param for max samples"
    )
    parser.add_argument(
        "--aim_repo", default="aim_gen", type=str, help="param for aim repo"
    )
    parser.add_argument("--cpu_metrics", action="store_true")
    parser.add_argument("--gpu_metrics", action="store_true")
    parser.add_argument(
        "--ref-seed",
        default=[1, 2],
        type=int,
        nargs="+",
        help="seed for ref metrics computation",
    )
    return parser


def make_mauve_parser():
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--device_id", default=0, type=int, help="param for device_id")
    parser.add_argument(
        "--batch_size", default=16, type=int, help="param for batch size"
    )
    parser.add_argument(
        "--feat-model", default="gpt2-large", type=str, help="param for model"
    )

    parser.add_argument("--layer", default="-1", help="param for layer")
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--recompute_ref", action="store_true")
    return parser


def get_generation_hparams(config):
    hparams = {}
    hparams["model"] = config.model.path
    hparams["nucleus_p"] = config.generation.get("top_p", 1.0)
    hparams["temperature"] = config.generation.get("temperature", 1.0)
    hparams["repetition_penalty"] = config.generation.repetition_penalty
    hparams["max_new_tokens"] = config.generation.max_new_tokens
    hparams["generation_seed"] = config.seed

    return hparams


def make_mauvepr_config():
    mauvepr = OmegaConf.create()
    mauvepr.seed = 1235
    mauvepr.pca = True
    return mauvepr


def make_prdc_config(args=None):
    prdc = OmegaConf.create()
    prdc.seed = 1235
    if args is not None:
        prdc.knn = args.knn
    prdc.pca = True
    return prdc


def get_mauvepr_prdc_hparams(eval_config):
    hparams = {}
    for k, v in eval_config.prdc.items():
        hparams[f"prdc_{k}"] = v

    for k, v in eval_config.mauvepr.items():
        hparams[f"mauvepr_{k}"] = v
    return hparams


def get_features_hparams(args):
    hparams = {}
    hparams["layer"] = args.layer
    hparams["feat_model"] = args.feat_model
    return hparams


def get_self_bleu_hparams(config):
    hparams = {}
    hparams[f"bleu-sample_ratio"] = config.sample_ratio_bleu
    hparams["bleu-tokenizer"] = config.tokenizer
    hparams["n_lst_bleu"] = config.n_lst_bleu
    hparams["n_lst"] = config.n_lst
    return hparams
