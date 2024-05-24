from pr_llm.evaluate.utils import get_generation_hparams


def get_hparams(config):
    hparams = {}
    hparams.update(get_generation_hparams(config))
    hparams["n_words"] = config.model.task.top_k_words
    return hparams
