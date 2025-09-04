import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

EPSILON = 1e-8
TARGET_TOLERANCE = 1E-5

def entropy(d_squared, beta):
    p = np.exp(-d_squared * beta)
    sum_p = np.sum(p)
    if sum_p == 0.0:
        sum_p = EPSILON
    sum_dp = np.sum(d_squared * p) / sum_p
    return np.log(sum_p) + beta * sum_dp

def cardinality(d, beta):
    return np.exp(-d * beta).sum()

def binary_search(dist, kernel, target, obj, n_steps):
    """计算有向边的权重"""
    if kernel == 'gaussian':
        dist = dist**2
    elif kernel == 'laplacian':
        dist -= dist[:,[0]]
    else:
        raise ValueError("kernel must be 'gaussian' or 'laplacian'")
        
    n_samples = dist.shape[0]
    betas = np.zeros(n_samples)

    for i in range(n_samples):
        beta_min =  0.0
        beta_max = np.inf
        beta = 1.0

        for n in range(n_steps):

            target_diff = obj(dist[i], beta) - target

            if np.fabs(target_diff) <= TARGET_TOLERANCE:
                break
            elif target_diff > 0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = 0.5 * (beta + beta_max)
            else:
                beta_max = beta
                beta = 0.5 * (beta_min + beta)

        betas[i] = beta
        
    if kernel == 'gaussian':
        p = np.exp(-dist * betas[:, None])
        p /= p.sum(axis=1)[:, None]
    elif kernel == 'laplacian':
        p = np.exp(-dist * betas[:, None])
        
    return p

def manifold_approximation(
            data,
            n_neighbors=15,
            kernel='laplacian',
            k=15,
            n_steps=64,
            symmetrize='union',
            verbose=False,
        ):
    
    # TODO: 检查kernel为gaussian时的情况

    # step 1: 无权有向
    neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    dist, ind = neigh.kneighbors()
    
    # step 2： 加权有向
    if kernel == 'gaussian':
        target = np.log2(k)
        obj = entropy
    elif kernel == 'laplacian':
        target = np.log2(k)
        obj = cardinality
    
    # p为有向边的权重
    p = binary_search(dist, kernel, target, obj, n_steps)
    n_samples = ind.shape[0]
    rows = (np.ones_like(ind) * np.arange(n_samples)[:, None]).ravel()
    P = coo_matrix((p.ravel(), (rows, ind.ravel())), shape=(n_samples, n_samples)).astype(np.float32)
    
    # step 3: 加权无向
    if symmetrize == 'mean':
        adm = 0.5 * (P + P.T)
    elif symmetrize == 'union':
        adm = P + P.T - P.multiply(P.T)
        
    return adm