from __future__ import division
from sklearn.metrics import ndcg_score
import numpy as np


def all_evaluations(y_true, y_score, k_list = [20,100,200]):
    """ Compute Recall@K, MAP@K, NDCG@K for all K in k_list
    :param y_true: ground-truth matrix
    :param y_score: prediction matrix
    :param k_list: list of K to consider
    """

    for k in k_list:

        # Compute scores @k
        recallk = recall_k(y_true, y_score, k)
        mapk = map_k(y_true, y_score, k)
        ndcgk = ndcg_score(y_true, y_score, k)

        # Print scores (rounded, and in percentage)
        print("Recall@{}: {}%".format(k, np.round(100*recallk, 3)), \
              "MAP@{}: {}%".format(k, np.round(100*mapk, 3)), \
              "NDCG@{}: {}%".format(k, np.round(100*ndcgk, 3)) )


# Disclaimer: the below functions for Recall@K and MAP@K computations
# are heavily inspired from: https://gist.github.com/mblondel/7337391

def recall_k(y_true, y_score, k = 10):
    """ Compute mean Recall@k on all individuals
    :param y_true: ground-truth matrix
    :param y_score: prediction matrix
    :param k: threshold k
    :return: Recall@k
    """

    nb_samples = y_true.shape[0]
    y_true = (y_true > 0.0)*1.0 # Binarize y_true
    r = []

    for i in range(nb_samples):
        precision, recall = ranking_precision_score(y_true[i,:], y_score[i,:], k)
        r.append(recall)

    return np.mean(r)


def ranking_precision_score(y_true, y_score, k = 10):
    """ Compute precision@k and recall@k for an individual
    :param y_true: ground-truth
    :param y_score: prediction
    :param k: threshold k
    :return: Precision and recall at rank k
    """

    unique_y = np.unique(y_true)
    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Get precision and recall
    return float(n_relevant) / min(n_pos, k), float(n_relevant) / n_pos


def map_k(y_true, y_score, k = 10):
    """ Compute mean AP@k on all individuals
    :param y_true: ground-truth matrix
    :param y_score: prediction matrix
    :param k: threshold k
    :return: MAP@k
    """

    y_true = (y_true > 0.0)*1.0 # Binarize y_true
    nb_samples = y_true.shape[0]
    map_k = []

    for i in range(nb_samples):
        # AP for user i
        map_k.append(ap_k(y_true[i,:], y_score[i,:],k))

    # MAP over users
    return np.mean(map_k)


def ap_k(y_true, y_score, k = 10):
    """ Compute ap@k for an individual
    :param y_true: ground-truth
    :param y_score: prediction
    :param k: threshold k
    :return: AP at rank k
    """

    unique_y = np.unique(y_true)
    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:k]
    y_true = np.asarray(y_true)[order]
    score = 0

    for i in range(k):
        if y_true[i] == pos_label:
            prec = np.sum(y_true[:(i+1)])/min(n_pos,i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos