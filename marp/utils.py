import numpy as np

def aggregate(x, nodes, method='mean'):
    """将原图的属性，聚合成抽象图的属性"""
    x = np.asarray(x)
    if method == 'mean':
        y = [np.mean(x[nodes[i]]) for i in range(len(nodes))]
    else:
        print('TODO')
    
    return y

def disperse():
    """将聚合图的属性，分散到原图"""
    pass