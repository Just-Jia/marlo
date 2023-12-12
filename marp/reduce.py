import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components

def reduction(data,
              graph,
              lens="PCA",
              lens_kwds={"n_components": 2},
              resolution=10,
              epsilon=1e-5):
    """约简流形近似图"""
    # Project through lens
    if lens == "PCA":
        r = PCA(**lens_kwds).fit_transform(data)
    elif lens == "self":
        r = data
    else:
        print('TODO:')

    # Find a codomain and grid
    n, d = r.shape

    # TODO: 修改取最大值和最小值的方式
    # new
    v_mins = r.min(axis=0)
    v_delta = (np.max(r.max(axis=0) - r.min(axis=0)) + 2 * epsilon) / resolution
    
    # old
#     v_mins = np.ones(d) * (r.min() - epsilon)
#     v_maxs = np.ones(d) * (r.max() + epsilon)
#     v_delta = (v_maxs - v_mins) / resolution

    # Put data into grid element
    base = np.array([np.power(resolution, i) for i in range(d)])
    grid = {}
    for i in range(n):
        # index of grid where point i located
        inds = np.floor((r[i,:] - v_mins) / v_delta).astype(np.int_)
        ind = str(np.dot(inds, base))    
        if ind in grid.keys():
            grid[ind].append(i)
        else:
            grid[ind] = [i]
    # convert list to np.array
    for key, value in grid.items(): 
        grid[key] = np.array(value)
    
    # Construct node
    nodes = []
    for key, points in grid.items():
        sub_graph = graph[np.ix_(points, points)]
        n_cc, labels = connected_components(csgraph=csr_matrix(sub_graph), directed=False, return_labels=True)
        for i in range(n_cc):
            nodes.append(points[np.where(labels==i)[0]])
            
    # Construct edge, i.e. adjacency matrix
    n_node = len(nodes)
    # index for each data point in the nodes
    ids=[-1 for i in range(n)]
    for i in range(n_node):
        for node in nodes[i]:
            ids[node] = i

    graph_triu = triu(graph, k=1)
    rs, cs = graph_triu.nonzero()

    # initialize adjacency matrix
    adjacency = np.zeros(shape=(n_node, n_node))
    for i in range(rs.shape[0]):
        row = ids[rs[i]]
        col = ids[cs[i]]
        if row != col:
            if graph[rs[i], cs[i]] > adjacency[row, col]:
                adjacency[col, row] = adjacency[row, col] = graph[rs[i], cs[i]]
    # convert to sparse matrix
    adjacency = csr_matrix(adjacency)
    return nodes, adjacency