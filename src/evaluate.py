import numpy as np


def precision_at_k(recommended_items, actual_items, k=10):
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(actual_items))
    return hits / k


def recall_at_k(recommended_items, actual_items, k=10):
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(actual_items))
    if len(actual_items) == 0:
        return 0
    return hits / len(actual_items)


def ndcg_at_k(recommended_items, actual_items, k=10):
    recommended_k = recommended_items[:k]

    dcg = 0
    for i, item in enumerate(recommended_k):
        if item in actual_items:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(actual_items), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0

    return dcg / idcg