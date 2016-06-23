__author__ = 'Trappola'

import networkx as nx

cut = 3
path = "actor_network_weighted_overlap.csv"
graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float), ('NOverlap', float)))

for o, d, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'], data['NOverlap'])):
        graph.remove_edge(o, d)
        if data["weight"] >= cut:
            break

subgraph_cut = max(nx.connected_component_subgraphs(graph), key=len)
out_file = open("actor_network_weighted_overlap_cut"+str(cut)+".csv", "w")
nx.write_edgelist(subgraph_cut, out_file, delimiter=",", data=False)
print len(subgraph_cut.nodes())
print len(subgraph_cut.edges())