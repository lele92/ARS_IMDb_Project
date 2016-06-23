__author__ = 'Trappola'

import networkx as nx
import json
import copy

cut = 3
path = "actor_network_weighted_overlap.csv"
graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float), ('NOverlap', float)))

path_actor = "../DATA/File_IMDb/actor_full_genre_cleaned.json"

file_actor = open(path_actor).read()
actors_data = json.loads(file_actor)

for o, d, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'], data['NOverlap'])):

        if data["weight"] >= cut:
            if not actors_data[o]["award"] or not actors_data[d]["award"]:
                # print str(len(actors_data[o]["award"])) +" "+ str(len(actors_data[d]["award"]))
                graph.remove_edge(o, d)
        else:
            graph.remove_edge(o, d)

# graph_copy = copy.deepcopy(graph)
# for component in nx.connected_component_subgraphs(graph_copy):
#     if len(component) == 1:
#         for node in component.nodes():
#             graph.remove_node(node)
#
# print nx.number_connected_components(graph)
# for component in nx.connected_component_subgraphs(graph):
#     print len(component)
subgraph_cut3 = max(nx.connected_component_subgraphs(graph), key=len)
# out_file = open("actor_network_weighted_overlap_cut"+str(cut)+"_awarded.csv", "w")
out_file = open("actor_network_cut"+str(cut)+"_awarded.csv", "w")
# nx.write_edgelist(subgraph_cut3, out_file, delimiter=",", data=('weight', 'NOverlap'))
nx.write_edgelist(subgraph_cut3, out_file, delimiter=",", data=False)
print len(subgraph_cut3.nodes())
print len(subgraph_cut3.edges())