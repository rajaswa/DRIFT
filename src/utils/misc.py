import json
import os

import numpy as np
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def list_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_sub(x, rev=False):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    if rev:
        res = x.maketrans("".join(sub_s), "".join(normal))
    else:
        res = x.maketrans("".join(normal), "".join(sub_s))
    return x.translate(res)


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans("".join(normal), "".join(super_s))
    return x.translate(res)


def get_tail_from_data_path(data_path):
    return os.path.split(data_path)[-1].split(".")[0]


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def reduce_dimensions(
    vectors, compass_vectors=None, typ="tsne", output_dimensions=2, fit_on_compass=False
):
    if fit_on_compass is True:
        if typ == "tsne":
            raise NotImplementedError(
                f"'tsne' type not supported when `fit_on_compass` is set to 'True'."
            )
        if compass_vectors is None:
            raise ValueError(
                f"`compass_vectors` cannot be of type: {type(compass_vectors)} when `fit_on_compass` is set to 'True'."
            )

    if typ == "pca":
        pca = PCA(output_dimensions, random_state=42)
        if fit_on_compass:
            compass_embeddings = pca.fit_transform(compass_vectors)
            embeddings = pca.transform(vectors)
        else:
            embeddings = pca.fit_transform(vectors)
    elif typ == "tsne":
        tsne = TSNE(n_components=output_dimensions, init="pca", random_state=42)
        # compass_embeddings = tsne.fit_transform(compass_vectors)
        embeddings = tsne.fit_transform(vectors)
    elif typ == "umap":
        reducer = umap.UMAP(
            n_components=output_dimensions, transform_seed=42, random_state=42
        )
        if fit_on_compass:
            compass_embeddings = reducer.fit_transform(compass_vectors)
            embeddings = reducer.transform(vectors)
        else:
            embeddings = reducer.fit_transform(vectors)
    else:
        raise NotImplementedError(f"No implementation found for `typ`: {typ}.")
    return embeddings


def save_json(dict_obj, save_path):
    if save_path is not None:
        with open(save_path, "w") as json_f:
            json.dump(dict_obj, json_f, indent=4)


def save_npy(arr, save_path):
    if save_path is not None:
        with open(save_path, "wb") as npy_f:
            np.save(npy_f, arr)


def remove_keywords_util(remove_keywords_path, sorted_gram_count_mapping):
    with open(remove_keywords_path, "r") as f:
        removed_keywords = f.read().split(",")
    sorted_gram_count_mapping = {
        key: sorted_gram_count_mapping[key]
        for key in sorted_gram_count_mapping.keys()
        if key not in removed_keywords
    }
    return sorted_gram_count_mapping


def length_removed_keywords(remove_keywords_path):
    with open(remove_keywords_path, "r") as f:
        len_keywords = len(f.read().split(","))
    return len_keywords
