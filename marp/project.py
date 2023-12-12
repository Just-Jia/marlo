import numpy as np
from sklearn.utils import check_random_state
from scipy.optimize import curve_fit
from sknetwork.embedding import Spectral
from numba import njit, prange

def projection(adm, 
               pos=None, 
               dim=2, 
               n_iters=500, 
               seed=None, 
               method='fruchterman_reingold', 
               method_kwds={}):
    """网络布局"""
    if method == 'fruchterman_reingold':
        pos = fruchterman_reingold(adm, pos=pos, dim=dim, n_iters=n_iters, seed=seed, **method_kwds)
    elif method == 'fruchterman_reingold_v1':
        pos = fruchterman_reingold_v1(adm, pos=pos, dim=dim, n_iters=n_iters, seed=seed, **method_kwds)
    elif method == 'spectral':
        pos = spectral(adm, dim=dim)
    elif method == 'random':
        pos = random(adm, dim=dim, seed=seed)
    elif method == 'umap':
        # TODO: add pos 
        pos = layout_umap(adm, dim=dim, seed=seed, n_epochs=n_iters, **method_kwds)
    else:
        print('TODO')
    
    return pos

# ---- random layout ----
def random(adm,
           dim=2,
           seed=None):
    """Graph layout by random"""
    
    random_state = check_random_state(seed)
    n_nodes = adm.shape[0]
    pos = random_state.rand(n_nodes, dim)
    
    return pos

# ---- spectral layout ----
def spectral(adm,
             dim=2):
    """Graph layout by graph spectral"""
    pos = Spectral(n_components=dim).fit_transform(adm)
    
    return pos

# ---- Fruchterman Reingold layout ----
def fruchterman_reingold(adm, 
                         dim=2, 
                         k=None, 
                         pos=None, 
                         n_iters=500, 
                         threshold=1e-4, 
                         seed=None):
    """Graph layout by Fruchtermen-Reingold force-directed method. referto networkX"""
    # prepare
    n_nodes = adm.shape[0]
    adm = adm.tolil()
    seed = check_random_state(seed)

    # initial pos
    if pos is None:
        pos = np.asarray(seed.rand(n_nodes, dim), dtype=adm.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1 / n_nodes)

    # cooling scheme
    t = max(pos.max(axis=0) - pos.min(axis=0)) * 0.1
    dt = t / (n_iters + 1)

    displacement = np.zeros((dim, n_nodes))


    for ite in range(n_iters):
        displacement *= 0

        for i in range(n_nodes):
            # difference between node i and others
            delta = (pos[i] - pos).T
            distance = np.sqrt((delta**2).sum(axis=0))

            # enforce minimum distance 
            distance = np.where(distance < 0.01, 0.01, distance)
            Ai = adm.getrowview(i).toarray()
            # displacement是位移在两个轴上的分量，f_attr =d^2 / k, f_repu = -k^2 / d
            displacement[:,i] += (delta * (k * k / distance**2 - Ai * distance / k)).sum(axis=1)
            # displacement为dim * n_nodes的矩阵，
            # disp[:,i]实际上表示所有节点的合力在两个轴上的分量

        # length 为1*n_nodes的矩阵，length[i]表示其它所有节点在节点i上的合力
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos

        # cooling
        t -= dt
        if (np.linalg.norm(delta_pos) / n_nodes) < threshold:
            break
            
    return pos

# ---- another version of Fr layout ----
@njit(fastmath=True)
def get_cols(adm_row, adm_col, i):
    """计算i对应的列"""
    return adm_col[np.where(adm_row==i)]

# @njit(parallel=True, fastmath=True) # parallel==True时会出错，不知道什么原因
@njit(fastmath=True)
def optimize_layout_fd(n_nodes,
                       pos,
                       adm_row,
                       adm_col,
                       k,
                       t,
                       n_sample):
    """使用力导向方法对布局进行一次优化"""

    for i in prange(n_nodes):
        delta = (pos[i] - pos).T
        ds = np.sqrt((delta**2).sum(axis=0))

        # 吸引力
#         inds = adm.rows[i]
        inds = get_cols(adm_row, adm_col, i)
        fa = (delta[:,inds] * ds[inds] / k).sum(axis=1)

        # 计算排斥力
        # 全部节点
        inds = np.concatenate((np.arange(i), np.arange(i + 1, n_nodes)))
        if n_sample is not None:
            inds = inds[np.random.randint(0, n_nodes - 1, n_sample)]
        fr = (delta[:, inds] * ( k ** 2 / ds[inds] ** 2)).sum(axis=1)

        f = fr - fa
        l = np.sqrt((f ** 2).sum(axis=0))
        pos[i] += (f * t / l).T
        

def fruchterman_reingold_v1(adm, 
                            dim=2, 
                            k=None, 
                            pos=None, 
                            init=None,
                            n_iters=500,
                            n_sample=None,
                            threshold=1e-4, 
                            dtype=np.float32,
                            seed=None):
    """Graph layout by Fruchtermen-Reingold force-directed method. referto networkX"""
    
    n_nodes = adm.shape[0]
    random_state = check_random_state(seed)

    # graph
    adm = adm.tocoo()
#     adm = adm.tolil()


    # initialize pos
    if pos is None:
        if init in [None, 'random']:
            pos = random_state.rand(n_nodes, dim).astype(dtype)
        elif init == 'spectral':
            pos = Spectral(n_components=dim).fit_transform(adm).astype(dtype)
            pos = (pos - pos.min(axis=0)) / (pos.max(axis=0) - pos.min(axis=0))

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1 / n_nodes)

    # cooling scheme
    t = max(pos.max(axis=0) - pos.min(axis=0)) * 0.1
    dt = t / (n_iters + 1)

    n_edges = adm.data.shape[0]
    adm_row = adm.row
    adm_col = adm.col

    for ite in range(n_iters):
        optimize_layout_fd(n_nodes,
                           pos,
                           adm_row,
                           adm_col,
                           k,
                           t,
                           n_sample)
        t -= dt
    return pos

# ---- UMAP layout ----
@njit
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val
    
  
def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def layout_umap(adm, 
                spread=1.,
                min_dist=0.1,
                n_epochs=500,
                seed = 1,
                dim = 2,
                alpha = 1., # 随机梯度学习率
                negative_sample_rate = 5, # 负样本率
                move_other = True,
                gamma = 1.):
    
    graph = adm
    
    a, b = find_ab_params(spread, min_dist)
    
    # 检测随机状态
    random_state = check_random_state(seed=seed)

    # 转化为coo
    graph = graph.tocoo()
    # 消除重复边
    graph.sum_duplicates()
    # 节点数
    n_vertices = graph.shape[1]

    # 消除权重太小的边，因为sampling机制下，这些边不被采到
    graph.data[graph.data < (1 / n_epochs)] = 0
    graph.eliminate_zeros()

    # 初始化布局
    initialisation = spectral(graph.tocsr())

    # 放缩到0-10,然后增加噪音
    expansion = 10.0 / np.abs(initialisation).max()
    embedding = (initialisation * expansion).astype(np.float32) + random_state.normal(scale=1e-4, size=(n_vertices, dim)).astype(np.float32)

    # 为sampling scheme 做准备
    # 吸引边的间隔
    epochs_per_sample = 1 / graph.data
    # 负样本的间隔（排斥力）
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    # 当前位置
    epoch_of_next_sample = epochs_per_sample.copy()
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()

    # 头，尾，权重
    head = graph.row
    tail = graph.col
    weight = graph.data

    # ？？？
    INT32_MIN = np.iinfo(np.int32).min + 1
    INT32_MAX = np.iinfo(np.int32).max - 1
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    aux_data = {}

    # 重新放缩到0-10，我觉得没必要，因为增加噪音后，初始布局不会改变很多
    embedding = 10. * (embedding - embedding.min(axis=0)) / (embedding.max(axis=0) - embedding.min(axis=0))
    embedding = embedding.astype(np.float32, order="C")

    n_edges = epochs_per_sample.shape[0]
    head = graph.row
    tail = graph.col
    weight = graph.data
    
    for n in range(n_epochs):
        optimize_layout(embedding, 
                        n,
                        n_vertices,
                        n_edges,
                        dim,
                        head,
                        tail,
                        weight,
                        epochs_per_sample,
                        epoch_of_next_sample,
                        epochs_per_negative_sample,
                        epoch_of_next_negative_sample,
                        a,
                        b,
                        gamma,
                        alpha,
                        move_other)

        # Todo: 降低学习率，参考alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
    return embedding

@njit(parallel=True, fastmath=True)
def optimize_layout(embedding, 
                    n,
                    n_vertices,
                    n_edges,
                    dim,
                    head,
                    tail,
                    weight,
                    epochs_per_sample,
                    epoch_of_next_sample,
                    epochs_per_negative_sample,
                    epoch_of_next_negative_sample,
                    a,
                    b,
                    gamma,
                    alpha=1,
                    move_other=True):
    """对布局进行优化"""
    
    # 对于每条边
    for i in prange(n_edges):
        # 如果当前的epoch 超过（大于或等于）当前的边epoch时，进行计算吸引力和排斥力
        if epoch_of_next_sample[i] <= n:
            # 计算吸引力和排斥力
            
            # 第 i 条边的头节点和尾节点
            j = head[i]
            k = tail[i]
            
            # 投节点和尾节点当前的位置
            current = embedding[j]
            other = embedding[k]

            # 头和尾节点距离的平方
            dist_squared = np.sum((current - other)**2)
            # 计算吸引力
            ### ??? 不清楚这里为什么是这个公式
            grad_coeff = - (2 * a * b * np.power(dist_squared, b - 1)) / (1 + a * np.power(dist_squared, b))
            # 移动两个节点
            for d in range(dim):
                # 计算位移，并且移动
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                
                if move_other:
                    other[d] += -grad_d * alpha
                    
            # 向后移动一个epoch间隔数
            epoch_of_next_sample[i] += epochs_per_sample[i]

            # 计算排斥力的次数
            n_neg_samples = int((n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
            
            # 任意选n_neg_samples个节点，然后计算排斥力
            for p in range(n_neg_samples):
                # 随机挑选一个节点
                k = np.random.randint(n_vertices)
                other = embedding[k]
                # 计算排斥力
                dist_squared = np.sum((current - other)**2)
                if dist_squared > 0:
                    # 计算排斥力
                    grad_coeff = (2.0 * gamma * b) / ((0.001 + dist_squared) * (1 + a * np.power(dist_squared, b)))

                    for d in range(dim):
                        # 计算位移，并且移动
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                        current[d] += grad_d * alpha
                else:
                    continue