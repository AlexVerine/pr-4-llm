import os
import random
import time
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.util import ngrams as ngrams_fn_nltk
from tqdm import tqdm
from transformers import AutoTokenizer

from pr_llm.evaluate.utils import make_same_length
from pr_llm.mauve import compute_mauve
from pr_llm.PR import compute_PR


class BaseScorer:
    # Same code than make-same-length but replace ref_split by p and current_split by q

    def score_exp(self, dataset, exp_fn: Callable, exp_args: List[Dict]):
        results = []
        for arg in exp_args:
            p, q = exp_fn(dataset=dataset, **arg)
            scorer_results = self.score(p, q)
            results.append(scorer_results)
        return results


class MAUVEPRScorer(BaseScorer):
    """Class for MAUVE and Sajjadi PR scoring."""

    def __init__(self, device_id=0, seed=0, pca=0.9, verbose=True):
        self.device_id = device_id
        self.seed = seed
        self.verbose = verbose
        if isinstance(pca, float):
            self.pca = pca
        elif isinstance(pca, bool):
            self.pca = 0.9 if pca else 1

    def __call__(self, p, q):
        p, q = make_same_length(p, q)
        mauve_pr_results = compute_mauve(
            p_features=p,
            q_features=q,
            device_id=self.device_id,
            featurize_model_name=None,
            seed=self.seed,
            verbose=self.verbose,
            kmeans_explained_var=self.pca,
        )
        metrics = {
            "mauvepr.mauve": mauve_pr_results.mauve,
            "mauvepr.precision": mauve_pr_results.precision,
            "mauvepr.recall": mauve_pr_results.recall,
            "mauvepr.fi_score": mauve_pr_results.frontier_integral,
        }
        artifacts = {
            "mauvepr.hist": mauve_pr_results.hist,
            "mauvepr.divergence_curve": mauve_pr_results.divergence_curve,
            # "mauvepr.num_buckets": mauve_pr_results.num_buckets,
            "mauvepr.prcurve": mauve_pr_results.prcurve,
        }

        return {"metrics": metrics, "artifacts": artifacts, "figures": {}}


class PRDCScorer(BaseScorer):
    """Class for Precision and Recall."""

    def __init__(
        self,
        pca=True,
        knn=4,
        device_id=0,
        seed=0,
        verbose=True,
    ):
        self.pca = pca
        self.knn = knn
        self.device_id = device_id
        self.seed = seed
        self.verbose = verbose

    def __call__(self, p, q):
        p, q = make_same_length(p, q)
        prdc_results = compute_PR(
            p_features=p,
            q_features=q,
            device_id=self.device_id,
            featurize_model_name=None,
            seed=self.seed,
            verbose=self.verbose,
            pca=self.pca,
            knn=self.knn,
        )

        metrics = {
            "prdc.precision": prdc_results.iP,
            "prdc.recall": prdc_results.iR,
            "prdc.density": prdc_results.D,
            "prdc.coverage": prdc_results.C,
        }
        return {"metrics": metrics, "artifacts": {}, "figures": {}}


class SelfBLEUDistinctNScorer:
    def __init__(
        self,
        n_lst=[1, 2, 3, 4],
        n_lst_bleu=[4],
        sample_ratio_bleu=0.2,
        tokenizer="gpt2",
        parallel_bleu=True,
        seed=12345,
    ):
        self.n_lst = n_lst
        self.n_lst_bleu = n_lst_bleu
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.sample_ratio_bleu = sample_ratio_bleu
        self.parallel_bleu = parallel_bleu
        self.rng = random.Random(seed)

    def get_ngram_freqs(self, samples, n):
        ngram_freq = Counter()
        for sen in samples:
            ngrams = ngrams_fn_nltk(sen, n)
            ngram_freq.update(ngrams)
        uniq = len(ngram_freq)
        total = sum(ngram_freq.values())
        return uniq, total

    def get_unique_ngram_fraction(self, samples, n_lst):
        # distinct-n
        results = {}
        for n in n_lst:
            a, b = self.get_ngram_freqs(samples, n)
            freq = a * 1.0 / b if b > 0 else 0
            results[f"distinct-{n}"] = freq
        return results

    def __call__(self, dataset, text_col):
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_col],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )

        dataset = dataset.map(
            tokenize_function,
            batched=True,
        )
        p_tok = dataset["input_ids"]
        results = self.get_unique_ngram_fraction(p_tok, self.n_lst)

        smoothing_function = SmoothingFunction().method1

        n_sample_bleu = int(len(dataset) * self.sample_ratio_bleu)

        start_time = time.time()
        if self.parallel_bleu:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            bleu_scores = self.compute_bleus_parallel(
                dataset["input_ids"],
                smoothing_function,
                self.rng,
                n_sample_bleu=n_sample_bleu,
                n_gram_lst=self.n_lst_bleu,
            )
        else:
            bleu_scores = self.compute_bleus_sequential(
                dataset["input_ids"],
                smoothing_function,
                self.rng,
                n_sample_bleu=n_sample_bleu,
                n_gram_lst=self.n_lst_bleu,
            )
        print("Total time for self bleu:", round(time.time() - start_time), "s")
        bleu_scores["n_samples"] = n_sample_bleu

        results.update(bleu_scores)

        return results

    def compute_bleus_sequential(
        self, all_sentences, smoothing_function, rng, n_sample_bleu, n_gram_lst
    ):
        bleu_scores = {}
        for n in n_gram_lst:
            start_time = time.time()
            weights = self.get_bleu_weight_for_ngram(n)
            bleu_n_lst = [
                self.self_bleu_one_sentence(
                    weights, all_sentences, smoothing_function, i
                )
                for i in rng.sample(
                    range(len(all_sentences)),
                    min(len(all_sentences), n_sample_bleu),
                )
            ]
            bleu_scores[f"bleu-{n}"] = sum(bleu_n_lst) / len(bleu_n_lst)
            print(
                f"Total time for self bleu-{n}:", round(time.time() - start_time), "s"
            )
        return bleu_scores

    def compute_bleus_parallel(
        self, all_sentences, smoothing_function, rng, n_sample_bleu, n_gram_lst
    ):
        pool = Pool(processes=os.cpu_count())
        print("Using", pool._processes, "processes for parallel bleu")
        bleu_scores = {}
        for n in n_gram_lst:
            start_time = time.time()
            weights = self.get_bleu_weight_for_ngram(n)
            bleu_n_lst = list(
                tqdm(
                    pool.imap_unordered(
                        partial(
                            self.self_bleu_one_sentence,
                            weights,
                            all_sentences,
                            smoothing_function,
                        ),
                        rng.sample(
                            range(len(all_sentences)),
                            min(len(all_sentences), n_sample_bleu),
                        ),
                    ),
                    total=n_sample_bleu,
                )
            )
            bleu_scores[f"bleu-{n}"] = sum(bleu_n_lst) / len(bleu_n_lst)
            print(
                f"Total time for self bleu-{n}:", round(time.time() - start_time), "s"
            )
        return bleu_scores

    def self_bleu_one_sentence(self, weights, all_sentences, smoothing_function, i):
        return sentence_bleu(
            references=all_sentences[:i] + all_sentences[i + 1 :],
            hypothesis=all_sentences[i],
            weights=weights,
            smoothing_function=smoothing_function,
        )

    def get_bleu_weight_for_ngram(self, n_gram):
        if n_gram == 1:
            weights = (1.0, 0, 0, 0)
        elif n_gram == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n_gram == 3:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
        elif n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 5:
            weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        else:
            raise ValueError
        return weights


def get_scorer(scorer_name: str, **kwargs):
    if scorer_name == "mauvepr":
        scorer = MAUVEPRScorer(**kwargs)
    elif scorer_name == "prdc":
        scorer = PRDCScorer(**kwargs)
    else:
        raise NotImplementedError
    return scorer


class MultiScorers(BaseScorer):
    def __init__(self, scorers_args: Dict):
        self.scorers_args = scorers_args
        self.list_scorers = []
        for scorer_name in self.scorers_args.keys():
            scorer = get_scorer(
                scorer_name=scorer_name, **self.scorers_args[scorer_name]
            )
            self.list_scorers.append(scorer)

    def __call__(self, p, q):
        results = {"metrics": {}, "artifacts": {}, "figures": {}}
        for scorer in self.list_scorers:
            scorer_results = scorer(p, q)
            for k in results.keys():
                results[k].update(scorer_results[k])
        return results
