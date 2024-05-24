import os
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from pr_llm.generation.preprocessing import get_data_collator
from pr_llm.utils import get_env


def get_dtype(type_str):
    if type_str == "bf16":
        return torch.bfloat16
    elif type_str == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_model_and_tokenizer(model_args, load_model=True):
    CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
    model_path = CHECKPOINT_PATH / model_args.path

    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = get_dtype(model_args.dtype)
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            pad_token_id=tokenizer.pad_token_id,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            device_map=model_args.device_map,
        )

    else:
        model = None
    return model, tokenizer


def get_dataset(data_path):
    DATA_PATH = get_env("DATA_PATH")
    return load_from_disk(str(DATA_PATH / data_path))


def load_model(config, load_model=True):
    model, tokenizer = get_model_and_tokenizer(config.model, load_model=load_model)

    data_collator = get_data_collator(
        tokenizer=tokenizer,
        **config.model.task,
    )

    return model, data_collator, tokenizer


class SavePathFormat:
    """
    Format the path for saving and loading the results.
    """

    def __init__(self, config, verbose=True):
        self.verbose = verbose
        self.config = config

    def get_tmp_path(self):
        TMP_PATH = get_env(os.environ["TMP_ENV"])
        return TMP_PATH / "tmp"

    def get_model_and_data_path(self):
        dataset_path = Path(self.config.data.path)
        model_name = Path(self.config.model.path)
        task_config = self.config.model.task
        generation_config = self.config.generation
        if task_config.task_name == "webtext":
            dataset_path = dataset_path / f"f{task_config.top_k_words}"
            if generation_config.do_sample:
                rep_pen = generation_config.get("repetition_penalty", 0)
                model_name = (
                    model_name
                    / f"topp{generation_config.top_p}_temp{generation_config.temperature}_rep{rep_pen}"
                )

        if task_config.task_name == "wikipedia_bio":
            dataset_path = dataset_path / f"ex{task_config.n_icl}"

            if generation_config.do_sample:
                model_name = (
                    model_name
                    / f"topp{generation_config.top_p}_temp{generation_config.temperature}_rep{rep_pen}"
                )
            else:
                model_name = model_name / f"greedy_rep{rep_pen}"

        return dataset_path, model_name

    def get_out_path(self):
        RESULT_PATH = get_env("RESULT_PATH")
        dataset_path, model_name = self.get_model_and_data_path()

        out_path = RESULT_PATH / dataset_path / model_name / "generation"
        if self.config.get("seed", None) is not None:
            out_path = out_path / f"seed{self.config.seed}"
        out_path.mkdir(exist_ok=True, parents=True)
        return out_path

    def get_save_path(self):
        out_path = self.get_out_path()
        current_time = datetime.now().strftime("%m-%d-%H:%M:%S")
        out_path = out_path / current_time
        out_path.mkdir(exist_ok=True, parents=True)
        return out_path

    def get_results_path(self, date="latest"):
        out_path = self.get_out_path()
        if self.verbose:
            print("Loading from:", out_path)

        if date == "latest":
            # Get latest timestamp in folder
            folders = [f for f in os.listdir(out_path)]
            timestamps_datetime = [
                datetime.strptime(ts, "%m-%d-%H:%M:%S") for ts in folders
            ]
            latest_folder = max(timestamps_datetime).strftime("%m-%d-%H:%M:%S")
        else:
            latest_folder = date

        out_path = out_path / latest_folder
        return out_path

    def get_generation_results_path(self, date="latest"):
        generation_path = self.get_results_path(date) / "predictions"
        if self.verbose:
            print("Loading from:", generation_path)
        return generation_path

    def get_evaluation_path(self, date="latest"):
        results_path = self.get_results_path(date)
        folder_name = results_path.name
        out_path = self.get_out_path()
        if self.config.get("seed", None) is not None:
            eval_path = (
                out_path.parent.parent / "evaluation" / f"seed{self.config.seed}"
            )
        eval_path = eval_path / folder_name
        if self.verbose:
            print("Eval_path:", eval_path)
        return eval_path

    def get_q_feat_name(self, feat_model, layer):
        return f"q_{feat_model}_{layer}"

    def get_q_feat_path(self, feat_model, layer, date="latest"):
        eval_path = self.get_evaluation_path(date=date)
        q_feat_name = self.get_q_feat_name(feat_model, layer)
        return eval_path / q_feat_name

    def get_pr_save_path(self, feat_model, layer, seed, date="latest", knn=None):
        q_feat_name = self.get_q_feat_name(feat_model, layer)
        eval_path = self.get_evaluation_path(date=date)
        if knn is not None:
            save_path = eval_path / f"{q_feat_name}_s{seed}_pr_knn{knn}.pkl"
        else:
            save_path = eval_path / f"{q_feat_name}_s{seed}_pr.pkl"

        return save_path
