from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizerBase


@dataclass
class WebTextFormatter:
    tokenizer: PreTrainedTokenizerBase
    top_k_words: int = 50
    max_length: int = 256
    chat_tokens: Optional[Tuple] = None

    def __call__(self, x):
        x_words = x["text"].split(" ")[: self.top_k_words]
        x_top_k_words = " ".join(x_words[: self.top_k_words])
        if self.chat_tokens is not None:
            x_words = x["text"].split(" ")
            x_top_k1_words = " ".join(x_words[: self.top_k_words - 1])
            x_top_k_words = f"{self.chat_tokens[0]} Continue the following text:\n\n{x_top_k1_words} {self.chat_tokens[1]} {x_words[self.top_k_words - 1]}"
        input_ids = self.tokenizer.encode(
            x_top_k_words,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = [self.tokenizer.bos_token_id] + input_ids

        token_types_instruction = [0] * len(input_ids)

        token_types = token_types_instruction

        output = {
            "input_ids": input_ids,
            "token_type_ids": token_types,
        }

        return output

    def format_batch(self, batch):
        return [self(x) for x in batch]


@dataclass
class CreativeFormatter:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 256
    min_essay_length: int = 100
    max_essay_length: int = 500
    chat_tokens: Optional[Tuple] = None

    def __call__(self, x):
        prompt = x["prompt"]
        prompt = f"{prompt} Your text should be approximately between {self.min_essay_length} and {self.max_essay_length} words."
        if self.chat_tokens is not None:
            prompt = f"{self.chat_tokens[0]} {prompt} {self.chat_tokens[1]} Sure here is a proposition:\n\n"
        else:
            prompt = f"Below is a text following the instructions:\n\n{prompt}\n\n### Text:\n\n"
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = [self.tokenizer.bos_token_id] + input_ids

        token_types_instruction = [0] * len(input_ids)

        token_types = token_types_instruction

        output = {
            "input_ids": input_ids,
            "token_type_ids": token_types,
        }

        return output

    def format_batch(self, batch):
        return [self(x) for x in batch]


@dataclass
class ICLFormatter:
    tokenizer: PreTrainedTokenizerBase
    str_header: str
    str_instruction: str
    max_length: int = 4096
    n_icl: int = 1
    chat_tokens: Optional[Tuple] = None

    def format_prompt_examples(self, x):
        examples = [x[f"example_{i}"] for i in range(self.n_icl)]
        prompt_str = self.str_header
        if self.chat_tokens is not None:
            prompt_str = f"{self.chat_tokens[0]} {prompt_str}"
        list_of_prompts = []
        for example in examples:
            title = example["title"]
            instruction = f"{self.str_instruction} {title}:\n\n"

            content = example["content"]

            prompt_string = f"{instruction}{content}"
            list_of_prompts.append(prompt_string)

        prompt_str += "\n\n".join(list_of_prompts)
        if self.chat_tokens is not None:
            prompt_str = f"{prompt_str} {self.chat_tokens[1]}"
        prompt_str += f"\n\n{self.str_instruction}"

        return prompt_str

    def __call__(self, x):
        prompt_str = self.format_prompt_examples(x)
        instruction_ids = self.tokenizer.encode(
            prompt_str,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,
        )
        instruction_ids = [self.tokenizer.bos_token_id] + instruction_ids

        token_types_instruction = [0] * len(instruction_ids)

        input_ids = instruction_ids

        token_types = token_types_instruction

        output = {
            "input_ids": input_ids,
            "token_type_ids": token_types,
        }

        return output

    def format_batch(self, batch):
        return [self(x) for x in batch]


@dataclass
class LLMDataCollator:
    tokenizer: PreTrainedTokenizerBase
    task_formater: Any
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def __call__(self, batch, return_tensors=None):
        features = self.task_formater.format_batch(batch)
        if return_tensors is None:
            return_tensors = self.return_tensors

        main_features = [
            {
                "input_ids": x["input_ids"],
                "token_type_ids": x["token_type_ids"],
            }
            for x in features
        ]
        main_features = self.tokenizer.pad(
            main_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return main_features


def get_data_collator(task_name, **kwargs):
    data_collator_kwargs = {
        "tokenizer": kwargs["tokenizer"],
        "max_length": kwargs["max_length"],
        "pad_to_multiple_of": kwargs.get("pad_to_multiple_of", None),
    }
    formater_kwargs = {
        "tokenizer": kwargs["tokenizer"],
        "max_length": kwargs["max_length"],
    }
    formater_kwargs["chat_tokens"] = kwargs.get("chat_tokens", None)

    if task_name == "webtext":
        formater = WebTextFormatter(
            top_k_words=kwargs["top_k_words"], **formater_kwargs
        )
        data_collator = LLMDataCollator(task_formater=formater, **data_collator_kwargs)
    elif task_name == "creative":
        formater = CreativeFormatter(
            min_essay_length=kwargs["min_essay_length"],
            max_essay_length=kwargs["max_essay_length"],
            **formater_kwargs,
        )
        data_collator = LLMDataCollator(
            task_formater=formater,
            **data_collator_kwargs,
        )

    elif task_name == "wikipedia_bio":
        formater = ICLFormatter(
            str_header=kwargs["str_header"],
            str_instruction=kwargs["str_instruction"],
            n_icl=kwargs["n_icl"],
            **formater_kwargs,
        )
        data_collator = LLMDataCollator(task_formater=formater, **data_collator_kwargs)

    else:
        raise ValueError(f"Unknown task: {task_name}")
    return data_collator
