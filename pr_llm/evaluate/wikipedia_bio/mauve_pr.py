import pickle as pkl
import re
from argparse import ArgumentParser
from functools import partial

import numpy as np
from aim import Run, Text
from datasets import load_from_disk
from dotenv import load_dotenv
from omegaconf import OmegaConf

from pr_llm.evaluate.scores import MultiScorers, SelfBLEUDistinctNScorer
from pr_llm.evaluate.utils import (
    get_args_layer,
    get_features_hparams,
    get_mauvepr_prdc_hparams,
    make_mauve_parser,
    make_mauvepr_config,
    make_prdc_config,
)
from pr_llm.evaluate.wikipedia_bio.eval import get_hparams, make_wikipedia_bio_parser
from pr_llm.generation.utils import SavePathFormat
from pr_llm.mauve.compute_mauve import get_features_from_input
from pr_llm.utils import get_env


def eval_wikipedia_bio(p, q, config):
    eval_config = config.evaluation
    scorer = MultiScorers({"mauvepr": eval_config.mauvepr, "prdc": eval_config.prdc})

    results = scorer(p, q)

    return results


def format_dataset(example):
    example["formated_text"] = re.split(
        r"\n\n- Biography|\n\n#", example["prediction"]
    )[0]

    return example


def format_ref_dataset(example):
    example["formated_text"] = f"{example['title']}:\n\n{example['content']}"

    return example


if __name__ == "__main__":
    mauve_parser = make_mauve_parser()
    wikipedia_parser = make_wikipedia_bio_parser()
    parser = ArgumentParser(parents=[mauve_parser, wikipedia_parser], add_help=False)
    parser.add_argument("--gen-seed", type=int, default=1)

    args = parser.parse_args()
    load_dotenv()
    CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
    DATA_PATH = get_env("DATA_PATH")

    # Define config
    config = OmegaConf.load(args.config)
    eval_config = OmegaConf.create()
    eval_config.mauvepr = make_mauvepr_config()
    eval_config.prdc = make_prdc_config(args)

    eval_config.ref_dataset_path = "wikipedia_bio"
    config.evaluation = eval_config

    args.layer = get_args_layer(args.layer)

    config.seed = args.gen_seed

    if args.gpu_metrics:
        for ref_seed in args.ref_seed:
            run = Run(experiment="wikipedia_bio", repo=args.aim_repo)

            hparams = {}
            hparams = get_hparams(config)
            hparams.update(get_features_hparams(args))
            hparams.update(get_mauvepr_prdc_hparams(eval_config))
            hparams["ref_seed"] = ref_seed
            for k, v in hparams.items():
                run[k] = v

            ref_dataset = load_from_disk(
                DATA_PATH / config.evaluation.ref_dataset_path,
            )

            ref_dataset = ref_dataset.shuffle(seed=ref_seed)
            ref_dataset = ref_dataset.select(range(args.max_samples))

            ref_dataset = ref_dataset.map(
                format_ref_dataset, load_from_cache_file=False
            )

            p_feature_path = (
                DATA_PATH
                / config.evaluation.ref_dataset_path
                / f"ref_features_{args.feat_model}_l{args.layer}_s{ref_seed}.pkl"
            )
            featurize = partial(
                get_features_from_input,
                device_id=args.device_id,
                batch_size=args.batch_size,
                layer=args.layer,
                featurize_model_name=str(CHECKPOINT_PATH / args.feat_model),
                features=None,
                tokenized_texts=None,
                max_len=1024,
            )
            if not p_feature_path.exists() or args.recompute_ref:
                p_features = featurize(texts=ref_dataset["formated_text"], name="ref")

                with open(p_feature_path, "wb") as f:
                    np.save(f, p_features)
            else:
                print("Loading ref features from cache")
                with open(p_feature_path, "rb") as f:
                    p_features = np.load(f)
            print("Ref features shape", p_features.shape)

            n_icl_list = args.n_icl
            for step, n_icl in enumerate(n_icl_list):
                try:
                    run.track(value=n_icl, name="n_icl", step=step)
                    config.model.task.n_icl = n_icl
                    save_format = SavePathFormat(config)
                    # Generation
                    generation_path = save_format.get_generation_results_path()
                    generation_dataset = load_from_disk(generation_path).select(
                        range(args.max_samples)
                    )

                    # Format generation to remove placeholder tokens
                    generation_dataset = generation_dataset.map(format_dataset)

                    # Print some samples
                    print("\n\n------\n\nGen:\n\n")
                    print(
                        "\n\n------\n\n".join(
                            [x[:200] for x in generation_dataset["formated_text"][:3]]
                        )
                    )
                    # Get features

                    evaluation_path = save_format.get_evaluation_path(args.date)

                    q_features_path = save_format.get_q_feat_path(
                        args.feat_model, layer=args.layer
                    )

                    if not q_features_path.exists() or args.recompute:
                        q_features = featurize(
                            texts=generation_dataset["formated_text"], name="q"
                        )
                        q_features_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(q_features_path, "wb") as f:
                            np.save(f, q_features)
                    else:
                        print("Loading ref features from cache")
                        with open(q_features_path, "rb") as f:
                            q_features = np.load(f)
                    print("Q features shape", q_features.shape)

                    save_path = save_format.get_pr_save_path(
                        feat_model=args.feat_model, layer=args.layer, seed=ref_seed
                    )
                    if save_path.exists():
                        print("Loading PR metrics from cache")
                        with open(save_path, "rb") as f:
                            results = pkl.load(f)
                    else:
                        results = eval_wikipedia_bio(p_features, q_features, config)
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(save_path, "wb") as f:
                            pkl.dump(results, f)
                    for k, v in results["metrics"].items():
                        run.track(value=v, name=k, step=step)
                    run.track(
                        name="save_path",
                        value=Text(str(save_path)),
                        step=step,
                    )
                except Exception as e:
                    print(f"Error with {args.config} n_icl {n_icl}", e)
                    run.track(
                        name="error",
                        value=Text(f"Error with n_icl {n_icl}"),
                        step=step,
                    )

    if args.cpu_metrics:
        hparams = {}
        hparams = get_hparams(config)
        hparams.update(get_features_hparams(args))
        hparams.update(get_mauvepr_prdc_hparams(eval_config))

        bleu_scorer = SelfBLEUDistinctNScorer()
        for n_icl in args.n_icl:
            config.model.task.n_icl = n_icl
            save_format = SavePathFormat(config)
            # Generation
            generation_path = save_format.get_generation_results_path()
            generation_dataset = load_from_disk(generation_path).select(
                range(args.max_samples)
            )

            # Format generation to remove placeholder tokens
            generation_dataset = generation_dataset.map(format_dataset)

            # Print some samples
            print("\n\n------\n\nGen:\n\n")
            print(
                "\n\n------\n\n".join(
                    [x[:200] for x in generation_dataset["formated_text"][:3]]
                )
            )
            # Get features

            evaluation_path = save_format.get_evaluation_path(args.date)
            save_path = evaluation_path / "cpu_metrics.pkl"
            if save_path.exists() and (not args.recompute_cpu):
                print("Loading CPU metrics from cache")
                with open(save_path, "rb") as f:
                    results = pkl.load(f)

            else:
                results = {}
                results_bleu = bleu_scorer(generation_dataset, "formated_text")
                results.update(results_bleu)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    pkl.dump(results, f)
