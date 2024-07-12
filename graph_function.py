import numpy as np
from scipy import sparse as sp

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def dopca(x, dim=50):
    pca = PCA(n_components=dim)
    pca.fit(x)
    return pca.transform(x)


def get_adj(count, k=15, pca=50, mode="connectivity"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def create_edge_index(x, k=5, pca_dim=50):
    x_p = dopca(x, dim=pca_dim)
    A = kneighbors_graph(x_p, k, mode='connectivity', metric="euclidean", include_self=True)
    A = A.toarray()
    A = np.float32(A)
    index = np.nonzero(A)
    e_index = np.concatenate((np.expand_dims(index[0], axis=0), np.expand_dims(index[1], axis=0)), axis=0)
    return e_index, A


