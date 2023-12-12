import numpy as np

from scipy.sparse import triu

import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot(adm, pos, method='plotly',
         cmap=plt.cm.rainbow,
         node_color=None,
         node_size=None,
         show_edges=True,
         show_labels=False,
         edge_color=None,
         edge_width=None,
         show_axis=False,
         width=600,
         height=400,
         xaxis_range=None,
         yaxis_range=None,
         margin=dict(b=0, l=0, r=0, t=20),
         verbose=0):
    """可视化网络"""
    
    n = len(pos)
    
    if method == 'matplotlib':
        fig, ax = plt.subplots(figsize=(8,6))

        # Plot edges
        if show_edges:
            edge_list = [(i,j) for i in range(n-1) for j in range(i+1, n) if adm[i,j]>0]
            edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edge_list])
            edge_collection = LineCollection(edge_pos, 
                                             colors=edge_color,
                                             linewidths=edge_width)
            edge_collection.set_zorder(1)  # edges go behind nodes
            ax.add_collection(edge_collection)

        # Plot nodes
        out = ax.scatter(pos[:,0],
                         pos[:,1],
                         s=node_size,
                         c=node_color)
        if not show_axis:
            ax.axis('off')
        
        return out
    elif method == 'plotly':
        
        # set parameters
        if node_color is None:
            node_color = 'blue'
            
        if node_size is None:
            node_size = 6
        
        if edge_color is None:
            edge_color = 'grey'
        
        if edge_width is None:
            edge_width = 0.5
            
        if show_labels:
            mode = 'markers+text'
        else:
            mode = 'markers'
        
        # node trace
        trace_node = go.Scatter(x=pos[:, 0],
                                y=pos[:, 1],
                                text=[str(i) for i in range(n)],
                                textposition='top center',
                                mode=mode,
                                name='Nodes',
                                marker=dict(color=node_color,
                                            size=node_size))
        if show_edges:
            # edge trace
            A = triu(adm)
            n_edges = len(A.row)
            edge_x = []
            edge_y = []
            for i in range(n_edges):
                x0, y0 = pos[A.row[i]]
                x1, y1 = pos[A.col[i]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            trace_edge = go.Scatter(x=edge_x, 
                                    y=edge_y,
                                    line=dict(width=edge_width, 
                                              color=edge_color),
                                    hoverinfo='none',
                                    mode='lines',
                                    name='Edges')
            
            data = [trace_edge, trace_node]
        else:
            data = [trace_node]
        
        # layout
        layout = go.Layout(width=width,
                           height=height,
                           margin=margin,
                           xaxis_range=xaxis_range,
                           yaxis_range=yaxis_range)
        fig = go.Figure(data=data, layout=layout)
        fig.show()
    else:
        print('TODO')
    