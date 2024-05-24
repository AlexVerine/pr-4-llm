import pickle as pkl
from argparse import ArgumentParser
from functools import partial

import numpy as np
from aim import Run, Text
from datasets import load_from_disk
from dotenv import load_dotenv
from omegaconf import OmegaConf

from pr_llm.evaluate.creative.eval import get_hparams, make_creative_parser
from pr_llm.evaluate.scores import MultiScorers, SelfBLEUDistinctNScorer
from pr_llm.evaluate.utils import (
    get_args_layer,
    make_mauve_parser,
    make_mauvepr_config,
    make_prdc_config,
)
from pr_llm.generation.utils import SavePathFormat
from pr_llm.mauve.compute_mauve import get_features_from_input
from pr_llm.utils import get_env


def eval_creative(p, q, eval_config):
    scorer = MultiScorers({"mauvepr": eval_config.mauvepr, "prdc": eval_config.prdc})

    results = scorer(p, q)

    return results


def format_generation_dataset(example):
    example["formated_text"] = " ".join(example["prediction"].split(" ")[:500])
    return example


if __name__ == "__main__":
    base_parser = make_creative_parser()
    mauve_parser = make_mauve_parser()

    parser = ArgumentParser(parents=[base_parser, mauve_parser], add_help=False)
    parser.add_argument("--gen-seed", type=int, default=1)

    args = parser.parse_args()
    load_dotenv()

    CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
    DATA_PATH = get_env("DATA_PATH")
    RESULT_PATH = get_env("RESULT_PATH")
    # Define config
    config_a = OmegaConf.load(args.config_a)
    config_b = OmegaConf.load(args.config_b)
    eval_config = OmegaConf.create()
    eval_config.mauvepr = make_mauvepr_config()
    eval_config.prdc = make_prdc_config(args)
    config_a.evaluation = eval_config
    config_b.evaluation = eval_config
    # Load data

    args.layer = get_args_layer(args.layer)
    config_a.seed = args.gen_seed

    if args.gpu_metrics:
        # Generation
        save_format_a = SavePathFormat(config_a)
        generation_path_a = save_format_a.get_generation_results_path()
        generation_dataset_a = load_from_disk(generation_path_a).select(
            range(args.max_samples)
        )
        generation_dataset_a = generation_dataset_a.map(format_generation_dataset)
        featurize = partial(
            get_features_from_input,
            device_id=args.device_id,
            batch_size=args.batch_size,
            layer=args.layer,
            featurize_model_name=str(CHECKPOINT_PATH / args.feat_model),
            features=None,
            tokenized_texts=None,
            max_len=4096,
        )
        a_feat_path = save_format_a.get_q_feat_path(
            feat_model=args.feat_model, layer=args.layer
        )
        # Print some samples
        print("\n\n------\n\nGen A:\n\n")
        print(
            "\n\n------\n\n".join(
                [x[:200] for x in generation_dataset_a["formated_text"][:3]]
            )
        )
        if not a_feat_path.exists() or args.recompute:
            a_features = featurize(
                texts=generation_dataset_a["formated_text"], name="a"
            )
            a_feat_path.parent.mkdir(parents=True, exist_ok=True)
            with open(a_feat_path, "wb") as f:
                np.save(f, a_features)
        else:
            print("Loading ref features from cache")
            with open(a_feat_path, "rb") as f:
                a_features = np.load(f)
        for seed_b in range(1, 6):
            config_b.seed = seed_b
            save_format_b = SavePathFormat(config_b)

            generation_path_b = save_format_b.get_generation_results_path()
            generation_dataset_b = load_from_disk(generation_path_b).select(
                range(args.max_samples)
            )

            # Format generation to remove placeholder tokens
            generation_dataset_b = generation_dataset_b.map(format_generation_dataset)

            print("\n\n------\n\nGen B:\n\n")
            print(
                "\n\n------\n\n".join(
                    [x[:200] for x in generation_dataset_b["formated_text"][:3]]
                )
            )
            b_feat_path = save_format_b.get_q_feat_path(
                feat_model=args.feat_model, layer=args.layer
            )

            print("A features shape", a_features.shape)

            if not b_feat_path.exists() or args.recompute:
                b_features = featurize(
                    texts=generation_dataset_b["formated_text"], name="b"
                )
                b_feat_path.parent.mkdir(parents=True, exist_ok=True)
                with open(b_feat_path, "wb") as f:
                    np.save(f, b_features)
            else:
                print("Loading ref features from cache")
                with open(b_feat_path, "rb") as f:
                    b_features = np.load(f)
            print("B features shape", b_features.shape)
            model_a_name = save_format_a.get_model_and_data_path()[1]
            model_b_name = save_format_b.get_model_and_data_path()[1]

            comparative_name = (
                f"{model_a_name}_s{args.gen_seed}_vs_{model_b_name}_s{seed_b}"
            )
            evaluation_path = RESULT_PATH / "creative" / comparative_name
            evaluation_path.mkdir(parents=True, exist_ok=True)
            save_path = evaluation_path / "mauvepr.pkl"
            if save_path.exists():
                continue
            else:
                results = eval_creative(a_features, b_features, eval_config)

                with open(save_path, "wb") as f:
                    pkl.dump(results, f)

    if args.cpu_metrics:
        run = Run(experiment="crea_cpu", repo=args.aim_repo)
        hparams = {}
        hparams = get_hparams(config_a)

        scorer = SelfBLEUDistinctNScorer()
        save_format = SavePathFormat(config_a)
        # Generation
        generation_path_a = save_format_a.get_generation_results_path()
        generation_dataset_a = load_from_disk(generation_path_a).select(
            range(args.max_samples)
        )
        generation_dataset_a = generation_dataset_a.map(format_generation_dataset)

        # Print some samples
        print("\n\n------\n\nGen:\n\n")
        print(
            "\n\n------\n\n".join(
                [x[:200] for x in generation_dataset_a["formated_text"][:3]]
            )
        )
        # Get features

        evaluation_path = save_format.get_evaluation_path(args.date)
        save_path = evaluation_path / "cpu_metrics.pkl"
        if save_path.exists() and not args.recompute:
            print("Loading BLEU from cache")
            with open(save_path, "rb") as f:
                results = pkl.load(f)
        else:
            results = scorer(generation_dataset_a, "formated_text")
            with open(save_path, "wb") as f:
                pkl.dump(results, f)
        for k, v in results.items():
            run.track(value=v, name=k)
        run.track(
            name="save_path",
            value=Text(str(save_path)),
        )
