import numpy as np
from scipy.optimize import curve_fit
from sklearn.utils import check_random_state

import numba
from tqdm.auto import tqdm

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

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

def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights of how much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result

@numba.njit()
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
    
@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.intp,
    },
)
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]

def optimize_layout_euclidean(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
#     densmap=False,
#     densmap_kwds=None,
    tqdm_kwds=None,
    move_other=False,
):

    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
        _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=parallel
    )


    if tqdm_kwds is None:
        tqdm_kwds = {}

    epochs_list = None
    embedding_list = []
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    for n in tqdm(range(n_epochs), **tqdm_kwds):

        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

        if epochs_list is not None and n in epochs_list:
            embedding_list.append(head_embedding.copy())

    # Add the last embedding to the list as well
    if epochs_list is not None:
        embedding_list.append(head_embedding.copy())

    return head_embedding if epochs_list is None else embedding_list

def _optimize_layout_euclidean_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):

    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
            
def simplicial_set_embedding(
#     data,
    graph,
    a,
    b,
    random_state,
    pos=None,
    n_components=2,
    initial_alpha=1.0,
    gamma=1.0,
    negative_sample_rate=5,
    n_epochs=500,
    init='random',
#     metric,
#     metric_kwds,
#     densmap,
#     densmap_kwds,
#     output_dens,
#     output_metric=dist.named_distances_with_gradients["euclidean"],
#     output_metric_kwds={},
    euclidean_output=True,
    parallel=True,
    verbose=False,
    tqdm_kwds=None,
):

    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    # For smaller datasets we can use more epochs
    if graph.shape[0] <= 10000:
        default_epochs = 500
    else:
        default_epochs = 200

    # Use more epochs for densMAP
#     if densmap:
#         default_epochs += 200

    if n_epochs is None:
        n_epochs = default_epochs

    # If n_epoch is a list, get the maximum epoch to reach
    n_epochs_max = max(n_epochs) if isinstance(n_epochs, list) else n_epochs

    if n_epochs_max > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

    graph.eliminate_zeros()

    # here, 初始话布局
    if pos is not None:
        embedding = pos
    elif isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    else:
        raise TypeError("please input proper 'pos' or 'init'")

    # ---- 开始嵌入
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs_max)

    head = graph.row
    tail = graph.col
    weight = graph.data

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

#     aux_data = {}

    embedding = (
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    # print(f' - euclidean_output: ', end='')
    if euclidean_output:
        # print('Yes')

        # print(f' - parallel = {parallel}')

        embedding = optimize_layout_euclidean(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            parallel=parallel,
            verbose=verbose,
#             densmap=densmap,
#             densmap_kwds=densmap_kwds,
            tqdm_kwds=tqdm_kwds,
            move_other=True,
        )

    return embedding

def layout_umap_v1(adm, 
                   dim=2, 
                   spread=1.0, 
                   min_dist=0.1,
                   pos=None, 
                   init='random', 
                   n_iters=None,
                   seed=None,
                   kw=dict()):
    """原始的umap布局"""
    
    a, b = find_ab_params(spread, min_dist)
    random_state = check_random_state(seed)

    pos = simplicial_set_embedding(graph=adm, 
                                   a=a, 
                                   b=b, 
                                   random_state=random_state,
                                   pos=pos,
                                   init='random',
                                   n_components=dim,
                                   n_epochs=n_iters,
                                   **kw)
    return pos