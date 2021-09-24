config = {
    # Proportion of artists in validation/test sets (for baselines)
    # By default, the bottom 20% ot artists in the dataset are "cold" artists,
    # the first 10% being the "test" artists and the last 10% being the
    # "validation" artists (hence 'prop_test': 10.)
    'prop_test': 10.,
    # For 'validation':
    # if True: all scores will be computed on validation set
    # if False: all scores will be computed on test set
    'validation': False,
    # Value(s) of K for Recall@K, MAP@K and NDCG@K evaluations
    # Default values are the ones used in the paper
    'K': [20, 100, 200],
    # Dimension of z_i vectors for node embedding methods under consideration
    'dimension': 32,
    # Value of "lambda" hyperparameter used to balance masses and
    # distances in the gravity-inspired predictions
    'lambda': 5,
    # Add epsilon to distances computations for numerical stability'
    'epsilon': 0.01,
    # Number of neighbors to re-rank for the two "K-NN + Popularity" and
    # "K-NN + In-degree" baselines
    'n_neighbors': 200
}