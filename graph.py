class Graph:
    """ Weighted Graph
    
    Used for representing manifold approximation of high-dimensional data, or representing 
    a reduced graph of manifold approximation.

    Parameters
    ----------
    level : int
        The level of the graph. 0 represents the manifold approximation of the original 
        high-dimensional data, 1 represents the reduction of a 0-level graph, 2 represents 
        the reduction of a 1-level graph, and so on.

    data : array, shape (n_nodes, n_dim)
        图的节点坐标。如果0-level，则为原始高维数据，如果1-level，则为node所表示数据点集的质心。

    adm : array, shape (n_nodes, n_nodes)
        Adjacency matrix of graph. 'adm' can be a sparse matrix of type 'coo'.

    pos : array, shape (n_nodes, n_components)
        Position of nodes in the embedding space.

    nodes : list, shape (n_nodes)
        节点列表，即，每个节点都为上一级图的节点子集。

    n_nodes : int
        number of nodes.
    
    n_dim : int
        dimension of high-dimensional space where data locates.

    n_components : int
        dimension of embedding space.
    """
    def __init__(
        self, 
        level=None, 
        data=None, 
        adm=None, 
        pos=None, 
        nodes=None, 
        n_nodes=None,
        n_dim=None,
        n_components=None
    ):
        self.level = level
        self.data = data
        self.adm = adm
        self.pos = pos
        self.nodes = nodes
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.n_components = n_components

    def __str__(self):
        return f"Graph(level={self.level}, n_nodes={self.n_nodes}, n_dim={self.n_dim}, n_components={self.n_components})"