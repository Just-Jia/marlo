import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components

def reduction(graph, 
              mapping="pca",
              mapping_kwargs=dict(n_components=2),
              margin=1e-5,
              resolution=10,
              connected=True,
              verbose=False):
    """Reduction of graph
    """
    
    data = graph.data
    adm = graph.adm
    
    # step 4: feature mapping
    # image under feature mapping
    if mapping == "pca":
        image = PCA(**mapping_kwargs).fit_transform(data)
    elif mapping == "identity":
        image = data
    elif mapping == "coordinate":
        kwds = dict(axes=[0,1])
        kwds.update(mapping_kwargs)
        image = data[:, kwds['axes']]
    else:
        raise ValueError("'mapping' should be 'pca', 'identity', 'coordinate'")

    # step 5: grid partitioning
    # 这里，每个网格元素都是个方块
    # build grid, buttom point, side length of grid element
    n, d = image.shape
    buttom = image.min(axis=0) - margin
    side = max(image.max(axis=0) - image.min(axis=0)) + 2 * margin
    side /= resolution


    # TODO: 检查ind是否可能超出最大整数
    # Put data into grid element
    base = np.array([pow(resolution, i) for i in range(d)])
    grid = {}
    for i in range(n):
        # index of grid where point i located
        # idx - index vector of point i
        # ind - string index of point i
        idx = np.floor((image[i,:] - buttom) / side).astype('int')
        ind = str(np.dot(idx, base))    
        if ind in grid.keys():
            grid[ind].append(i)
        else:
            grid[ind] = [i]
    # convert list to np.array
    for key, value in grid.items(): 
        grid[key] = np.array(value)

    # step 6: Construct node
    # TODO: 检查邻接矩阵中的0值是否表示无边相连
    nodes = []
    if connected:
        for points in grid.values():
            sub_graph = adm[np.ix_(points, points)]
            n_cc, labels = connected_components(csgraph=csr_matrix(sub_graph), directed=False, return_labels=True)
            for i in range(n_cc):
                nodes.append(points[np.where(labels==i)[0]])
    else:
        for points in grid.values():
            nodes.append(points)

    # Construct edge, i.e. adjacency matrix
    n_nodes = len(nodes)
    # index for each data point in the nodes
    ids = np.zeros(n, dtype=int)
    for i in range(n_nodes):
        ids[nodes[i]] = i

    # All edge in old graph
    rs, cs = triu(adm, k=1).nonzero()

    # Initial adjacency matrix for abstract graph
    # 在新的抽象图中，节点与节点连边的权重为原图中两个节点间的权重的最大值
    new_adm = np.zeros(shape=(n_nodes, n_nodes))
    for r, c in zip(rs, cs):
        new_r = ids[r]
        new_c = ids[c]    
        if (new_r != new_c) and (new_adm[new_r, new_c] < adm[r, c]):
            new_adm[new_r, new_c] = new_adm[new_c, new_r] = adm[r, c]
    # convert to sparse matrix
    new_adm = csr_matrix(new_adm)

    # Construct new data
    new_data = np.zeros((n_nodes, data.shape[1]))
    for i in range(n_nodes):
        new_data[i,] = data[nodes[i],].mean(axis=0)
        
    return new_data, new_adm, nodes 