#%%
import numpy as np
import pandas as pd
import pytest
from graspologic.embed import CovariateAssistedEmbedding as CASE
from graspologic.simulations import sbm
from graspologic.plot import heatmap, pairplot, pairplot_with_gmm
from tests.test_casc import gen_covariates
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns

#%%
def gen_covariates(labels, m1=0.8, m2=0.2, agreement=1, d=3):
    """
    n x 3 matrix of covariates

    """
    N = len(labels)
    d = 3
    B = np.full((d, d), m2)
    B[np.diag_indices_from(B)] = m1
    base = np.eye(d)
    membership = np.zeros((N, d))
    n_misassign = 0
    for i in range(0, N):
        assign = bool(np.random.binomial(1, agreement))
        if assign:
            membership[i, :] = base[labels[i], :]
        else:
            membership[i, :] = base[(labels[i] + 1) % (max(labels) + 1), :]
            n_misassign += 1

    probs = membership @ B

    covariates = np.zeros(probs.shape)
    for i in range(N):
        for j in range(d):
            covariates[i, j] = np.random.binomial(1, probs[i, j])

    return covariates


def get_misclustering(A, model, labels, covariates=None) -> float:
    if covariates is None:
        Xhat = model.fit_transform(A)
    else:
        Xhat = model.fit_transform(A, covariates=covariates)

    kmeans = KMeans(n_clusters=3)
    labels_ = kmeans.fit_predict(Xhat)

    # to account for nonidentifiability
    labels_ = remap_labels(labels, labels_)
    misclustering = np.count_nonzero(labels - labels_) / len(labels)

    return misclustering


from graspologic.simulations import sbm
from graspologic.utils import remap_labels
from graspologic.plot import pairplot
from graspologic.embed import CovariateAssistedEmbedding

n = 500
assortative = False
p, q = 0.03, 0.015
if not assortative:
    p, q = q, p
A, labels = sbm(
    [n, n, n],
    p=[[p, q, q], [q, p, q], [q, q, p]],
    return_labels=True,
)
X = gen_covariates(labels, m1=0.8, m2=0.2, agreement=0.0)
X = gen_covariates(labels, m1=0.8, m2=0.2, agreement=1)
import seaborn as sns


case = CovariateAssistedEmbedding(
    n_components=3, embedding_alg="non-assortative", n_iter=10
)
case.fit(A, covariates=X, labels=labels)
Xhat = case.latent_left_
pairplot(Xhat, labels=labels)


# def M():
#     # module scope ensures that A and labels will always match
#     # since they exist in separate functions

#     # parameters
#     n = 100
#     p, q = 0.9, 0.3

#     # block probability matirx
#     P = np.full((2, 2), p)
#     P[np.diag_indices_from(P)] = q

#     # generate sbm
#     directed = False
#     return sbm([n] * 2, P, directed=directed, return_labels=True)


# def X(M):
#     _, labels = M
#     m1, m2 = 0.8, 0.3
#     return gen_covariates(m1, m2, labels, type="many")


# M = M()
# X = X(M)
# A, labels = M
# case = CASE(assortative=False)
# case.fit(A, X)
# latent = case.latent_left_
# latent
# # separate into communities
# df = pd.DataFrame(
#     {
#         "Type": labels,
#         "Dimension 1": latent[:, 0],
#         "Dimension 2": latent[:, 1],
#     }
# )

# # Average per-group
# means = df.groupby("Type").mean()


# # train a GMM, compare with true labels
# predicted = GaussianMixture(n_components=2).fit_predict(latent)


# # avg_dist_within = np.diag(pairwise_distances(means, oos_left))
# avg_dist_between = np.diag(pairwise_distances(means, oos_right))
# self.assertTrue(all(avg_dist_within < avg_dist_between))