__author__ = 'Trappola'

import networkx as nx
import sys

path = "../DATA/Network_data_final/actor_network_weighted.csv"
graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float),))

count = 0

for o, d, data in graph.edges(data=True):
    neighbors_o = graph.neighbors(o)
    neighbors_d = graph.neighbors(d)
    neighbors_intersection = set(neighbors_o).intersection(neighbors_d)
    neighbors_union = set(neighbors_o).union(neighbors_d)
    node_set = set()
    node_set.add(o)
    node_set.add(d)
    neighbors_union = neighbors_union - node_set
    neighbors_overlap_coefficient = float(len(neighbors_intersection))/float(len(neighbors_union))
    # if int(neighbors_overlap_coefficient) == 1:
    #     print o
    #     print d
    #     print neighbors_union
    #     print neighbors_intersection
    data["NOverlap"] = neighbors_overlap_coefficient
    if count % 1000 == 0:
        print count
    count += 1

for a, b, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'], data['NOverlap'])): # reverse=True per ordinare al contrario
    print data

