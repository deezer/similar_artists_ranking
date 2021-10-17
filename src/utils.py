import networkx as nx
import numpy as np
import scipy.sparse as sp


def load_deezer_data():
    """ Load Deezer's graph of artists, and their features
    :return: Graph adjacency matrix 'adj',
             Feature matrix 'features',
             Number of nodes 'num_nodes'
    """

    # Load directed graph dataset
    print("Loading Deezer's graph of artists")
    adj = nx.adjacency_matrix(nx.read_weighted_edgelist("../data/deezer_graph.csv",
                                                           delimiter=',',
                                                           create_using=nx.DiGraph(),
                                                           nodetype =float))
    adj = adj - sp.eye(adj.shape[0])
    num_nodes = adj.shape[0]
    print("... Successfully loaded Deezer's graph with", num_nodes, "artists")

    # Load artist features
    print("Loading artists' descriptive features")
    features = np.genfromtxt("../data/deezer_features.csv", delimiter=",")[:,1:]
    print("... Successfully loaded artists' features matrix, of dimension: ", features.shape)

    return adj, features, num_nodes


def load_embedding(model):
    """ Load pre-computed node embedding vectors
    :param model: model name
    :return: num_nodes*dimension embedding matrix
    """

    emb = np.genfromtxt("../embeddings/{}.csv".format(model), delimiter=",")

    return emb


def train_test_split(adj, prop_test):
    """ Preprocessing on input data
    :param adj: Graph adjacency matrix
    :param prop_test: 'prop_test'% of nodes will be placed in test set
    and 'prop_test'% of nodes in also be placed in validation set
    :return: Training graph adjacency matrix 'adj_train',
             Number of training nodes 'num_nodes_train',
             Number of test nodes 'num_nodes_test'
    """

    # Data splitting
    print("\nSplitting data in train/validation/test sets:")
    num_nodes = adj.shape[0]
    num_nodes_train = int(np.floor(0.01*(100. - 2*prop_test)*num_nodes))
    num_nodes_test = np.floor(0.5*(num_nodes-num_nodes_train)).astype(int)
    num_nodes_validation = num_nodes - num_nodes_train - num_nodes_test
    if num_nodes_train <= 0:
        raise ValueError("No artist in train set! Please decrease and prop_test")

    # Train data
    adj_train = adj[0:num_nodes_train, 0:num_nodes_train]

    # Print information
    print("...", num_nodes_train, "nodes in training set")
    num_nodes_test = np.floor(0.5*(num_nodes-num_nodes_train)).astype(int)
    num_nodes_validation = num_nodes - num_nodes_train - num_nodes_test
    print("...", num_nodes_validation, "nodes in validation set")
    print("...", num_nodes_test, "nodes in test set")

    return adj_train, num_nodes_train, num_nodes_test


def get_ground_truth(adj, num_nodes_train, num_nodes_test, validation):
    """ Retrieving validation/test ground-truth
    :param adj: Graph adjacency matrix
    :param num_nodes_train: Number of training nodes
    :param num_nodes_test: Number of test nodes
    :validation: if True, we will retrieve ids and ground-truth
    of validation set. If False: ids and ground-truth of test set.
    :return: Ids of validation/test nodes 'cold_ids',
             Neighbors of validation/test nodes 'ground_truth'
    """

    if validation:
        print("\nValidation set will be used for model evaluation")
        cold_ids = range(num_nodes_train + num_nodes_test, adj.shape[0])
        ground_truth = adj[cold_ids,:].toarray()
        print("... Retrieved ground-truth for validation set")
    else:
        print("\nTest set will be used for model evaluation")
        cold_ids = range(num_nodes_train, num_nodes_train + num_nodes_test)
        ground_truth = adj[cold_ids,:].toarray()
        print("... Retrieved ground-truth for test set")

    return cold_ids, ground_truth


def sigmoid(x):
    """ Sigmoid activation function
    :param x: scalar value
    :return: sigmoid activation
    """
    return 1 / (1 + np.exp(-x))