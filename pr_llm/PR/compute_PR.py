import time
from types import SimpleNamespace

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

try:
    import torch

    FOUND_TORCH = True
except (ImportError, ModuleNotFoundError):
    FOUND_TORCH = False

try:
    import transformers

    FOUND_TRANSFORMERS = True
except (ImportError, ModuleNotFoundError):
    FOUND_TRANSFORMERS = False

if FOUND_TORCH and FOUND_TRANSFORMERS:
    # only needed for tokenizing
    from .utils import (
        featurize_tokens_from_model,
        get_device_from_arg,
        get_model,
        get_tokenizer,
    )


MODEL, TOKENIZER, MODEL_NAME = None, None, None


def compute_PR(
    p_features=None,
    q_features=None,
    p_tokens=None,
    q_tokens=None,
    p_text=None,
    q_text=None,
    pca=True,
    pca_max_data=-1,
    kmeans_explained_var=0.9,
    featurize_model_name="gpt2-large",
    device_id=-1,
    max_text_length=1024,
    knn=5,
    verbose=False,
    seed=25,
    batch_size=1,
    use_float64=False,
):
    """
    Compute the MAUVE score between two text generations P and Q.

    P is either specified as ``p_features``, ``p_tokens``, or ``p_text``. Same with Q.

    :param ``p_features``: ``numpy.ndarray`` of shape (n, d), where n is the number of generations.
    :param ``q_features``: ``numpy.ndarray`` of shape (n, d), where n is the number of generations.
    :param ``p_tokens``: list of length n, each entry is torch.LongTensor of shape (1, length).
    :param ``q_tokens``: list of length n, each entry is torch.LongTensor of shape (1, length).
    :param ``p_text``: list of length n, each entry is a string.
    :param ``q_text``: list of length n, each entry is a string.
    :param ``num_buckets``: the size of the histogram to quantize P and Q. Options: ``'auto'`` (default, which is n/10) or an integer.
    :param ``pca_max_data``: the number data points to use for PCA. If `-1`, use all the data. Default -1.
    :param ``kmeans_explained_var``: amount of variance of the data to keep in dimensionality reduction by PCA. Default 0.9.
    :param ``kmeans_num_redo``: number of times to redo k-means clustering (the best objective is kept). Default 5.
        Try reducing this to 1 in order to reduce running time.
    :param ``kmeans_max_iter``: maximum number of k-means iterations. Default 500.
        Try reducing this to 100 in order to reduce running time.
    :param ``featurize_model_name``: name of the model from which features are obtained. Default 'gpt2-large'.
        We support all models which can be loaded from ``transformers.AutoModel.from_pretrained(featurize_model_name)``.
    :param ``device_id``: Device for featurization. Supply gpu_id (e.g. 0 or 3) to use GPU or -1 to use CPU.
    :param ``max_text_length``: maximum number of tokens to consider. Default 1024.
    :param ``divergence_curve_discretization_size``: Number of points to consider on the divergence curve. Default 25.
        Larger values do not offer much of a difference.
    :param ``mauve_scaling_factor``: The constant``c`` from the paper. Default 5.
        See `Best Practices <index.html#best-practices-for-mauve>`_ for details.
    :param ``verbose``: If True, print running time updates.
    :param ``seed``: random seed to initialize k-means cluster assignments.
    :param ``batch_size``: Batch size for feature extraction.
        A larger batch size speeds up computation.
        You might have to experiment to find the largest batch size that fits in your GPU memory.
        See `here <https://github.com/krishnap25/mauve/issues/8#issuecomment-1082075240>`_ for details.

    :return: an object with fields p_hist, q_hist, divergence_curve and mauve.

    * ``out.mauve`` is a number between 0 and 1, the MAUVE score. Higher values means P is closer to Q.
    * ``out.frontier_integral``, a number between 0 and 1. Lower values mean that P is closer to Q.
    * ``out.p_hist`` is the obtained histogram for P. Same for ``out.q_hist``.
    * ``out.divergence_curve`` contains the points in the divergence curve. It is of shape (m, 2), where m is ``divergence_curve_discretization_size``

    """

    if p_features is None and p_tokens is None and p_text is None:
        raise ValueError("Supply at least one of p_features, p_tokens, p_text")
    if q_features is None and q_tokens is None and q_text is None:
        raise ValueError("Supply at least one of q_features, q_tokens, q_text")
    p_features = get_features_from_input(
        p_features,
        p_tokens,
        p_text,
        featurize_model_name,
        max_text_length,
        device_id,
        name="p",
        verbose=verbose,
        batch_size=batch_size,
        use_float64=use_float64,
    )
    q_features = get_features_from_input(
        q_features,
        q_tokens,
        q_text,
        featurize_model_name,
        max_text_length,
        device_id,
        name="q",
        verbose=verbose,
        batch_size=batch_size,
        use_float64=use_float64,
    )

    # Acutal binning
    t1 = time.time()
    if pca:
        p_features, q_features = PCA_(
            p_features,
            q_features,
            whiten=False,
            pca_max_data=pca_max_data,
            explained_variance=kmeans_explained_var,
            seed=seed,
            verbose=verbose,
        )

    res_prdc = compute_prdc(p_features, q_features, nearest_k=knn)
    t2 = time.time()
    if verbose:
        print("total PRDC time:", round(t1 - t1, 2), "seconds")

    to_return = SimpleNamespace(
        iP=res_prdc["precision"],
        iR=res_prdc["recall"],
        D=res_prdc["density"],
        C=res_prdc["coverage"],
    )
    return to_return


def get_features_from_input(
    features,
    tokenized_texts,
    texts,
    featurize_model_name,
    max_len,
    device_id,
    name,
    batch_size,
    verbose=False,
    use_float64=False,
):
    global MODEL, TOKENIZER, MODEL_NAME
    if features is None:
        # Featurizing is necessary. Make sure the required packages are available
        if not FOUND_TORCH:
            raise ModuleNotFoundError(
                """PyTorch not found. Please install PyTorch if you would like to use the featurization.
                    For details, see `https://github.com/krishnap25/mauve` 
                    and `https://pytorch.org/get-started/locally/`.
                """
            )
        if not FOUND_TRANSFORMERS:
            raise ModuleNotFoundError(
                """Transformers not found. Please install Transformers if you would like to use the featurization.
                    For details, see `https://github.com/krishnap25/mauve` 
                    and `https://huggingface.co/transformers/installation.html`.
                """
            )

        if tokenized_texts is None:
            # tokenize texts
            if TOKENIZER is None or MODEL_NAME != featurize_model_name:
                if verbose:
                    print("Loading tokenizer")
                TOKENIZER = get_tokenizer(featurize_model_name)
            if verbose:
                print("Tokenizing text...")
            tokenized_texts = [
                TOKENIZER.encode(
                    sen, return_tensors="pt", truncation=True, max_length=max_len
                )
                for sen in texts
            ]
        # use tokenized_texts to featurize
        if TOKENIZER is None or MODEL_NAME != featurize_model_name:
            if verbose:
                print("Loading tokenizer")
            TOKENIZER = get_tokenizer(featurize_model_name)
        if MODEL is None or MODEL_NAME != featurize_model_name:
            if verbose:
                print("Loading model")
            MODEL = get_model(featurize_model_name, TOKENIZER, device_id)
            MODEL_NAME = featurize_model_name
        else:
            MODEL = MODEL.to(get_device_from_arg(device_id))
        if use_float64:
            MODEL = MODEL.double()
        if verbose:
            print("Featurizing tokens")
        features = (
            featurize_tokens_from_model(MODEL, tokenized_texts, batch_size, name)
            .detach()
            .cpu()
            .numpy()
        )
    else:
        features = np.asarray(features)
    return features


def PCA_(
    p,
    q,
    whiten=False,
    pca_max_data=-1,
    explained_variance=0.9,
    seed=0,
    verbose=False,
):
    assert 0 < explained_variance < 1
    if verbose:
        print(f"seed = {seed}")
    data1 = np.vstack([q, p])
    data1 = normalize(data1, norm="l2", axis=1)
    pca = PCA(n_components=None, whiten=whiten, random_state=seed + 1)
    if pca_max_data < 0 or pca_max_data >= data1.shape[0]:
        pca.fit(data1)
    elif 0 < pca_max_data < data1.shape[0]:
        rng = np.random.RandomState(seed + 5)
        idxs = rng.choice(data1.shape[0], size=pca_max_data, replace=False)
        pca.fit(data1[idxs])
    else:
        raise ValueError(
            f"Invalid argument pca_max_data={pca_max_data} with {data1.shape[0]} datapoints"
        )
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    if verbose:
        print(f"performing clustering in lower dimension = {idx}")
    data1 = pca.transform(data1)[:, : idx + 1]
    # Cluster
    data1 = data1.astype(np.float32)

    q_pca = data1[: len(q)]
    p_pca = data1[len(q) :]
    if verbose:
        print(f"From dimension {p.shape} to {p_pca.shape}")
    return p_pca, q_pca


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = pairwise_distances(data_x, data_y, metric="euclidean", n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print(
        "Num real: {} Num fake: {}".format(
            real_features.shape[0], fake_features.shape[0]
        )
    )

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k
    )
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k
    )
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (
        (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1))
        .any(axis=0)
        .mean()
    )

    recall = (
        (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0))
        .any(axis=1)
        .mean()
    )

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall, density=density, coverage=coverage)
