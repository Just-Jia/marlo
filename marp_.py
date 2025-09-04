from manifold import manifold_approximation
from reduce import reduction
from project import projection
from graph import Graph

class MARP:
    """Manifold Approximation, Reduction and Projection (MARP)

    MARP 是一种拓扑数据分析方法，主要用来提取并展示高维数据的拓扑结构。

    Parameters
    ----------
    """

    def __init__(
        self,
        mode='marp',
        n_neighbors=15,
        kernel='laplacian',
        k=15,
        n_steps=64,
        symmetrize='union',
        mapping='pca',
        mapping_kwargs=dict(n_components=2),
        resolution=10,
        margin=1e-5,
        connectivity=True,
        pos=None,
        init='random',
        n_components=2,
        n_iters=None,
        random_state=None,
        project_mode='adm',
        project_method='umap',
        project_kwargs=dict(),
        verbose=False
        ):
        
        # defaulf kwargs for different modes
        # TODO: umap 和umap_like的默认init参数改为spectral
        # TODO: 移除tsne_like和umap_like模式
        default_kwargs = {
            'mode':          ['marp',      'tsne',      'umap'],
            'n_neighbors':   [15,          None,         14],
            'kernel':        ['laplacian', 'gaussian',   'laplacian'],
            'k':             [None,        30,            None],
            'n_steps':       [64,          100,          64],
            'symmetrize':    ['union',     'mean',       'union'],
            'mapping':       ['pca',       'pca',          None],
            'resolution':    [10,          10,              None],
            'connectivity':  [True,        True,           None],
            'init':          ['random',    'pca',          'random'],
            'project_mode':  ['adm',       'data',        'data'],
            'project_method':['umap',      'tsne',        'umap'],
        }

        # initialize parameters
        idx_mode = default_kwargs['mode'].index(mode)
        self.mode = mode
        self.n_neighbors = default_kwargs['n_neighbors'][idx_mode]
        self.kernel = default_kwargs['kernel'][idx_mode]
        self.k = default_kwargs['k'][idx_mode]
        self.n_steps = default_kwargs['n_steps'][idx_mode]
        self.symmetrize = default_kwargs['symmetrize'][idx_mode]
        self.mapping = default_kwargs['mapping'][idx_mode]
        self.resolution = default_kwargs['resolution'][idx_mode]
        self.connectivity = default_kwargs['connectivity'][idx_mode]
        self.init = default_kwargs['init'][idx_mode]
        self.project_mode = default_kwargs['project_mode'][idx_mode]
        self.project_method = default_kwargs['project_method'][idx_mode]

        # update parameters
        # for manifold approximation
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.k = k
        self.n_steps = n_steps
        self.symmetrize = symmetrize

        # for reduction
        self.mapping = mapping
        self.mapping_kwargs = mapping_kwargs
        self.resolution = resolution
        self.margin = margin
        self.connectivity = connectivity

        # for projection
        self.pos = pos
        self.init = init
        self.n_components = n_components
        self.n_iters = n_iters
        self.random_state = random_state
        self.project_mode = project_mode
        self.project_method = project_method
        self.project_kwargs = project_kwargs
        self.verbose = verbose

    def check_params(self):
        """Check parameters"""
        # TODO: 设置tsne, tsne_like, umap, umap_like模式下的邻居数
        # TODO: 检查k和邻居数之间的关系
        pass

    def fit(self, X):
        """Fit MARP, namely, compute the abstract graph and its layout.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        self.n_smaples = X.shape[0]
        self.check_params()

        # Phase 1: Manifold Approximation
        self.g = Graph(level=0, data=X)
        self.g.adm = manifold_approximation(
            data=X,
            n_neighbors=self.n_neighbors,
            kernel=self.kernel,
            k=self.k,
            n_steps=self.n_steps,
            symmetrize=self.symmetrize,
            verbose=self.verbose
        )

        # Phase 2: Reduction
        self.h = Graph(level=1)
        self.h.data, self.h.adm, self.h.nodes = reduction(
            graph=self.g,
            mapping=self.mapping,
            mapping_kwargs=self.mapping_kwargs,
            margin=self.margin,
            resolution=self.resolution,
            connected=self.connectivity,
            verbose=self.verbose
        )

        # Phase 3: Projection
        self.h.pos = projection(
            graph=self.h,
            pos=self.pos,
            init=self.init,
            n_components=self.n_components,
            n_iters=self.n_iters,
            random_state=self.random_state,
            project_mode=self.project_mode,
            method=self.project_method,
            method_kwargs=self.project_kwargs,
            verbose=self.verbose
        )

        return self

    def fit_transform(self, X):
        """Fit MARP and transform X into a abstract graph with low-dimensional layout.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            
        Returns
        -------
        h : Graph object
            Returns the abstract graph with low-dimensional layout.
        """
        self.fit(X)
        return self.h