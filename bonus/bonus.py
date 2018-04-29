#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:13:37 2018

@author: YiChen
"""

import networkx as nx


#def example_graph():
#    """
#    Create the example graph from class. Used for testing.
#    Do not modify.
#    """
#    g = nx.Graph()
#    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F'),('A','E'),('G','C'),('B','E')])
#    nx.draw(g,with_labels=True)
#    plt.show()
#    return g


def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined in bonus.md.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    ###TODO
    neighbors = set(graph.neighbors(node))
    scores = []
    for n in graph.nodes():
        if n != node and not graph.has_edge(n, node):
            neighbors2 = set(graph.neighbors(n))
            intersect=sum(1/len(set(graph.neighbors(i))) for i in neighbors&neighbors2)
            union=sum(1/len(set(graph.neighbors(i))) for i in neighbors)+sum(1/len(set(graph.neighbors(i))) for i in neighbors2)
            scores.append(((node,n),intersect/union))
              
    return sorted(scores, key=lambda x: x[1], reverse=True)

    pass