from __future__ import division
from __future__ import print_function
from evaluation import all_evaluations
from options import config
from ranking import *
from utils import *
import networkx as nx
import numpy as np
import os
import scipy.sparse as sp


print("\n\nDATA PREPROCESSING\n")

# Load Deezer's graph of artists, and their features
adj, features, num_nodes = load_deezer_data()

# Preprocessing on input data
adj_train, num_nodes_train, num_nodes_test = train_test_split(adj, config['prop_test'])
# Retrieving ids of validation (if config['validation'] is True) or
# test (if config['validation'] is False) artists, as well as their
# ground truth similar artists
cold_ids, ground_truth = get_ground_truth(adj, num_nodes_train, num_nodes_test,
                                          config['validation'])


print("\n\nEVALUATING MODELS \n")

# Display information on evaluation
if config['validation']:
    print("Starting evaluations on validation set")
else:
    print("Starting evaluations on test set")
print("Recall@K, MAP@K and NDCG@K will be computed for K =", config['K'])


# Popularity-based methods
print("\nPopularity:")
preds = popularity(cold_ids, num_nodes)
all_evaluations(ground_truth, preds, config['K'])

print("\nPopularity by country:")
preds = popularity_by_country(cold_ids, num_nodes, features)
all_evaluations(ground_truth, preds, config['K'])

print("\nIn Degree:")
preds = in_degree(cold_ids, num_nodes, adj_train)
all_evaluations(ground_truth, preds, config['K'])

print("\nIn Degree by country:")
preds = in_degree_country(cold_ids, num_nodes, adj_train, features)
all_evaluations(ground_truth, preds, config['K'])


# Features-based methods
print("\nK-NN in input features space:")
preds = knn(cold_ids,features)
all_evaluations(ground_truth, preds, config['K'])

print("\nK-NN + Popularity re-ranking:")
preds = knn_reorder(cold_ids, num_nodes, adj_train, features,
                    rerank = "popularity")
all_evaluations(ground_truth, preds, config['K'])

print("\nK-NN + In-Degree re-ranking:")
preds = knn_reorder(cold_ids, num_nodes, adj_train, features,
                    rerank = "in-degree")
all_evaluations(ground_truth, preds, k_list = config['K'])


# Node embedding methods
# Predictions are computed from embedding vectors provided in
# the "embeddings" folder, and trained on Deezer usage data of
# the top 80% artists of the dataset.

# We therefore re-computed the evaluation set in case
# config['prop_test'] != 10. (corresponding to a 80%/10%/10% split)
# to avoid evaluating these pre-computed embeddings on train data
if config['prop_test'] != 10.:
    print("Warning: embedding learned from a '80%/10%/10%' artists split")
    adj_train, num_nodes_train, num_nodes_test = train_test_split(adj, 10.)
    cold_ids, ground_truth = get_ground_truth(adj, num_nodes_train, num_nodes_test,
                                              config['validation'])

print("\nSTAR-GCN:")
emb = load_embedding(model = "stargcn")
preds = emb_to_predictions(cold_ids, emb, method = 'knn')
all_evaluations(ground_truth, preds, config['K'])

print("\nDEAL:")
emb = load_embedding(model = "deal")
preds = emb_to_predictions(cold_ids, emb, method = 'knn')
all_evaluations(ground_truth, preds, config['K'])

print("\nDropoutNet:")
emb = load_embedding(model = "dropoutnet")
preds = emb_to_predictions(cold_ids, emb, method = 'knn')
all_evaluations(ground_truth, preds, config['K'])

print("\nSVD+DNN:")
emb = load_embedding(model = "svd")
preds = emb_to_predictions(cold_ids, emb, method = 'inner-product')
all_evaluations(ground_truth, preds, config['K'])


# Graph Autoencoders (To do)
print("\nGraph AE/VAE:\nto be added.")