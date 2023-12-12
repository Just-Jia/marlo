from .manifold import *
from .reduce import *
from .project import *

class Graph:
    def __init__(self, adm=None, pos=None, level=0, nodes=None, n_nodes=None):
        self.adm = adm
        self.pos = pos
        self.level = level
        self.nodes = nodes
        self.n_nodes = n_nodes
        
class MARP:
    def __init__(
        self, 
        mode='abstract',
        n_level=1,
        ma_method='UMA',
        ma_method_kwds=dict(),
        lens='PCA',
        lens_kwds=dict(n_components=2),
        resolution=10,
        epsilon=1e-5,
        proj_method='fruchterman_reingold_v1',
        proj_method_kwds=dict(),
        verbose=True,
    ):
        self.mode = mode
        self.n_level=n_level
        self.ma_method = ma_method
        self.ma_method_kwds = ma_method_kwds
        self.lens = lens
        self.lens_kwds = lens_kwds
        self.resolution = resolution
        self.epsilon = epsilon
        self.proj_method = proj_method
        self.proj_method_kwds = proj_method_kwds
        self.verbose = verbose
        
    def check_params(self):
        """检查参数的有效性"""
        pass
    
    def fit_transform(self, X):
        """进行嵌入或者抽象模式"""
        G0 = manifold_approximation(data=X, 
                                    method=self.ma_method, 
                                    method_kwds=self.ma_method_kwds)
        
        nodes, G1 = reduction(data=X, 
                              graph=G0, 
                              lens=self.lens, 
                              lens_kwds=self.lens_kwds, 
                              resolution=self.resolution)
        
        pos = projection(G1, 
                         method=self.proj_method,
                         method_kwds=self.proj_method_kwds)
        return G1, pos
    
    def abstract(self, X):
        """抽象模式"""
        G0 = Graph(level=0)
        G0.adm = manifold_approximation(data=X, 
                                        method=self.ma_method, 
                                        method_kwds=self.ma_method_kwds)
        
        G1 = Graph(level=1)
        G1.nodes, G1.adm = reduction(data=X, 
                                     graph=G0.adm, 
                                     lens=self.lens, 
                                     lens_kwds=self.lens_kwds, 
                                     resolution=self.resolution)
        
        G1.pos = projection(G1.adm, 
                            method=self.proj_method,
                            method_kwds=self.proj_method_kwds)
        
        return G1.adm, G1.nodes,  G1.pos
    
    def embedding(self, X):
        pass