import numpy as np
from options import config
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import distance_matrix
from utils import sigmoid


def popularity(cold_ids, num_nodes):
    """ Popularity-based method
    :param cold_ids: Ids of evaluated nodes
    :return: Prediction of similar neighbors 'preds'
    """

    # As training nodes are ordered by popularity, this method
    #simply consists in recommending the first K nodes
    preds = np.ones((len(cold_ids),num_nodes))
    preds = (num_nodes - np.array(range(num_nodes)))*preds

    return preds


def popularity_by_country(cold_ids, num_nodes, features):
    """ Popularity by country
    :param cold_ids: Ids of evaluated nodes
    :param num_nodes: Total number of nodes
    :param features: Node features matrix
    :return: Prediction of similar neighbors 'preds'
    """

    # Get nodes from same country
    country_onehot = features[:,32:52]
    same_country = (country_onehot.dot(np.transpose(country_onehot)))[cold_ids,:]

    # Popularity-based reordering
    pop = popularity(cold_ids, num_nodes)
    preds = np.multiply(pop, same_country)

    return preds


def in_degree(cold_ids, num_nodes, adj):
    """ In-Degree
    :param cold_ids: Ids of evaluated nodes
    :param num_nodes: Total number of nodes
    :param adj: Graph (train) adjacency matrix
    :return: Prediction of similar neighbors 'preds'
    """

    # Node-level "in degree" scores (and 0 for non-train nodes)
    deg = np.append(np.array(list(np.sum(adj,0).flat)), np.repeat(0.0,num_nodes - adj.shape[0]))
    preds = deg*np.ones((len(cold_ids),num_nodes))

    return preds


def in_degree_country(cold_ids, num_nodes, adj, features):
    """ In-Degree by country
    :param cold_ids: Ids of evaluated nodes
    :param num_nodes: Total number of nodes
    :param adj: Graph (train) adjacency matrix
    :param features: Node features matrix
    :return: Prediction of similar neighbors 'preds'
    """

    # Node-level "in degree" scores
    deg = in_degree(cold_ids, num_nodes, adj)

    # Get nodes from same country
    country_onehot = features[:,32:52]
    same_country = (country_onehot.dot(np.transpose(country_onehot)))[cold_ids,:]

    # Predict highest "in-degree" nodes from the same country
    preds = np.multiply(deg, same_country)

    return preds


def knn(cold_ids, features):
    """ K-NN on descriptive feature vectors
    :param cold_ids: Ids of evaluated nodesx
    :param features: Node features matrix
    :return: Prediction of similar neighbors 'preds'
    """

    # Get closest nodes in terms of 'features'
    preds = - pairwise_distances(features[cold_ids,:], features, metric='euclidean')

    return preds


def knn_reorder(cold_ids, num_nodes, adj, features, rerank = "popularity", n_neighbors = config['n_neighbors']):
    """ K-NN + Popularity or In-Degree re-ordering
    :param cold_ids: Ids of evaluated nodes
    :param num_nodes: Total number of nodes
    :param adj: Graph (train) adjacency matrix
    :param features: Node features matrix
    :param re-rank: 'popularity' or 'in-degree'
    :param n_neighbors: Number of neighbors to re-rank
    :return: Prediction of similar neighbors 'preds'
    """

    # Get 'n_neighbors' nearest neighbors in terms of 'features'
    knn_matrix = kneighbors_graph(features, n_neighbors = n_neighbors, mode = 'connectivity', \
                                  include_self = False, metric = 'euclidean').toarray()[cold_ids,:]

    # Re-ranking them
    if rerank == "popularity":
        pop = popularity(cold_ids, num_nodes)
    else:  # In-degree re-ranking
        pop = in_degree(cold_ids, num_nodes, adj)
    preds = np.multiply(pop, knn_matrix)

    return preds


def emb_to_predictions(cold_ids, emb, method = 'inner-product', d = config['dimension'],
                       lamb = config['lambda'], epsilon = config['epsilon']):
    """ Compute edge predictions from embedding vectors
    :param cold_ids: Ids of evaluated nodes
    :param emb: Pre-computed embedding matrix
    :param method: 'inner-product', 'knn', 'source-target' or 'gravity'
    :param d: Dimension of z_i embedding vectors
    :param lamb, epsilon: parameters for gravity-inspired decoding
    :return: Prediction of similar neighbors 'preds'
    """

    if method == 'inner-product':
        preds = sigmoid(emb.dot(emb.T) - np.eye(emb.shape[0]))[cold_ids,:]

    elif method == "knn":
        preds = - pairwise_distances(emb[cold_ids,:d], emb[:,:d], metric='euclidean')

    elif method == 'source-target':
        assert d % 2 == 0 # Check that d is even
        emb_source = emb[:,:int(d/2)] # Source part of z_i
        emb_target = emb[:,int(d/2):] # Target part of z_i
        preds = sigmoid(emb_source.dot(emb_target.T) - np.eye(emb_source.shape[0]))[cold_ids,:]

    elif method == 'gravity':

        # Masses = last column of embedding matrix
        mass = emb[:,(d-1)]*np.ones((len(cold_ids),emb.shape[0]))
        # z_i vectors: first (d-1) columns
        emb_only = emb[:,:(d-1)]

        dist = np.square(epsilon + distance_matrix(emb_only, emb_only))[cold_ids,:]
        preds = sigmoid(mass - lamb*np.log(dist))

    return preds