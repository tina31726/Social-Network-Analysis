#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:13:37 2018
@author: YiChen
"""

import networkx as nx



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
            if len(neighbors)!=0 and len(neighbors2)!=0:
                intersect=sum(1/len(set(graph.neighbors(i))) for i in neighbors&neighbors2)
                union=1/sum(len(set(graph.neighbors(i))) for i in neighbors)+1/sum(len(set(graph.neighbors(i))) for i in neighbors2)
                scores.append(((node,n),intersect/union))
              
    return sorted(scores, key=lambda x: x[1], reverse=True)

    pass
