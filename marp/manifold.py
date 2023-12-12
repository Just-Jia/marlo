import umap

def manifold_approximation(data,
                           method="UMA",
                           method_kwds={},
                           verbose=False):
    """流形近似"""
    if method == "UMA":
        graph = umap.UMAP(transform_mode="graph", **method_kwds).fit_transform(data)
        
    return graph