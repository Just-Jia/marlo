from project_umap import layout_umap_v1

def projection(
    graph,
    pos=None,
    init="random",
    n_components=2,
    n_iters=None,
    random_state=None,
    project_mode="adm",
    method="umap",
    method_kwargs={},
    verbose=False
):
    """投射函数，将图数据投射到低维空间中

    将流形近似或约简后的结果投射到地维空间中，以便于可视化和分析。

    Parameters:
    -----------
    graph: Graph类实例
        图数据结构

    pos: numpy.ndarray, shape=(n_nodes, n_components), optional, default=None
        初始节点位置，如果为None，则根据init参数进行初始化

    init: str, optional, default="random"
        节点位置初始化方法，若project_mode为"data"，则init参数必须为"pca", "random"等。
        若project_mode为"adm"，则init参数必须为"spectral","random"等。

    n_components: int, optional, default=2
        降维后的维度

    n_iters: int, optional, default=None
        迭代次数，若为None，则根据数据量、投射模式、投射方法自动设置

    random_state: int, optional, default=None
        随机数种子

    project_mode: str, optional, default="adm"
        节点位置投射模式，"data"表示将图数据投射到低维空间，"adm"表示将邻接矩阵投射到低维空间。

    method: str, optional, default="umap"
        节点位置投射方法，"tsne", "umap", "spectral"等。

    method_kwargs: dict, optional, default={}
        节点位置投射方法参数
        
    verbose: bool, optional, default=False
        是否显示日志信息

    Returns:
    --------
    pos: numpy.ndarray, shape=(n_nodes, n_components)
        节点位置
    """
    # TODO: 检查project_mode参数为data时的情况
    # TODO: 添加更多的投射方法
    if project_mode == "data":
        pos = project_data(
            data=graph.data,
            pos=pos if pos is not None else init,
            n_components=n_components,
            n_iters=n_iters,
            random_state=random_state,
            method=method,
            **method_kwargs,
            verbose=verbose
        )
    elif project_mode == "adm":
        pos = project_adm(
            adm=graph.adm,
            pos=pos,
            init=init,
            n_components=n_components,
            n_iters=n_iters,
            random_state=random_state,
            method=method,
            **method_kwargs,
            verbose=verbose
        )
    else:
        raise ValueError("project_mode must be 'data' or 'adm'")
    return pos 


# 定义数据投射函数
def project_data(
        data,
        pos=None,
        n_components=2,
        n_iters=None,
        random_state=None,
        method='umap',
        verbose=False,
        **kwargs
):
    """将数据投射到一个低维空间中"""
    if method == 'umap':
        if pos is not None:
            pos = umap.UMAP(
                init=pos,        
                n_components=n_components, 
                n_epochs=n_iters, 
                random_state=random_state,
                **kwargs
            ).fit_transform(data)
        else:
            pos = umap.UMAP(
                n_components=n_components, 
                n_epochs=n_iters, 
                random_state=random_state,
                **kwargs
            ).fit_transform(data)
    else:
        raise ValueError(f"不支持的投射方法{method}")

    return pos

# 定义邻接矩阵投射函数
def project_adm(
        adm,
        pos=None,
        init='random',
        n_components=2,
        n_iters=None,
        random_state=None,
        method='umap',
        verbose=False,
        **kwargs
):
    """将图数据投射到一个低维空间中"""
    if method == 'umap':
        pos = layout_umap_v1(
            adm=adm,
            dim=n_components,
            pos=pos,
            init=init,
            n_iters=n_iters,
            seed=random_state,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的投射方法{method}")
    
    return pos