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
    make_eval_parser,
    make_mauve_parser,
    make_mauvepr_config,
    make_prdc_config,
)
from pr_llm.evaluate.webtext.eval import get_hparams
from pr_llm.generation.utils import SavePathFormat
from pr_llm.mauve.compute_mauve import get_features_from_input
from pr_llm.utils import get_env


def eval_webtext_pr(p, q, config):
    eval_config = config.evaluation
    scorer = MultiScorers({"mauvepr": eval_config.mauvepr, "prdc": eval_config.prdc})
    results = scorer(p, q)
    return results


def format_generation_dataset(example, top_k_words=10):
    pred = (
        f"{' '.join(example['text'].split(' ')[:top_k_words])} {example['prediction']}"
    )
    pred = " ".join(pred.split(" "))[:370]

    return {"formated_text": pred}


def format_ref_dataset(x):
    return {"formated_text": " ".join(x["text"].split(" ")[:370])}


if __name__ == "__main__":
    mauve_parser = make_mauve_parser()
    base_parser = make_eval_parser()

    parser = ArgumentParser(parents=[base_parser, mauve_parser], add_help=False)
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
    eval_config.ref_dataset_path = "webtext_15k"
    config.evaluation = eval_config

    args.layer = get_args_layer(args.layer)

    config.seed = args.gen_seed

    if args.gpu_metrics:
        for ref_seed in args.ref_seed:
            run = Run(experiment="webtext", repo=args.aim_repo)

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
            if not p_feature_path.exists() or args.recompute:
                p_features = featurize(texts=ref_dataset["formated_text"], name="ref")

                with open(p_feature_path, "wb") as f:
                    np.save(f, p_features)
            else:
                print("Loading ref features from cache")
                with open(p_feature_path, "rb") as f:
                    p_features = np.load(f)
            print("Ref features shape", p_features.shape)

            save_format = SavePathFormat(config, verbose=True)
            # Generation
            generation_path = save_format.get_generation_results_path()
            generation_dataset = load_from_disk(generation_path).select(
                range(args.max_samples)
            )

            # Format generation to remove placeholder tokens
            generation_dataset = generation_dataset.map(
                format_generation_dataset,
                fn_kwargs={"top_k_words": config.model.task.top_k_words},
            )

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
                feat_model=args.feat_model,
                layer=args.layer,
                seed=ref_seed,
                knn=config.evaluation.prdc.knn,
            )
            if save_path.exists() and not args.recompute_gpu:
                print("Loading PR from cache")
                with open(save_path, "rb") as f:
                    results = pkl.load(f)

            else:
                results = eval_webtext_pr(p_features, q_features, config)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    pkl.dump(results, f)
            for k, v in results["metrics"].items():
                run.track(value=v, name=k)
            run.track(
                name="save_path",
                value=Text(str(save_path)),
            )

    if args.cpu_metrics:
        run = Run(experiment="diversity_cpu", repo=args.aim_repo)
        hparams = {}
        hparams = get_hparams(config)

        scorer = SelfBLEUDistinctNScorer()
        save_format = SavePathFormat(config)
        # Generation
        generation_path = save_format.get_generation_results_path()
        generation_dataset = load_from_disk(generation_path).select(
            range(args.max_samples)
        )

        # Format generation to remove placeholder tokens
        generation_dataset = generation_dataset.map(
            format_generation_dataset,
            fn_kwargs={"top_k_words": config.model.task.top_k_words},
        )

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
        if save_path.exists() and not args.recompute_cpu:
            print("Loading BLEU from cache")
            with open(save_path, "rb") as f:
                results = pkl.load(f)
        else:
            results = scorer(generation_dataset, "formated_text")
            with open(save_path, "wb") as f:
                pkl.dump(results, f)
        for k, v in results.items():
            run.track(value=v, name=k)
        run.track(
            name="save_path",
            value=Text(str(save_path)),
        )
