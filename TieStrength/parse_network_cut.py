# -*- coding: utf-8 -*-
__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

import networkx as nx
import json

cut = 4
path = "actor_network_weighted_overlap.csv"
graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float), ('NOverlap', float)))

path_actor = "../DATA/File_IMDb/actor_full_genre_cleaned.json"

file_actor = open(path_actor).read()
actors_data = json.loads(file_actor)

for o, d, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'])):

        if data["weight"] >= cut:
            break
            # condition if I want to consider only awarded actor
            # if not actors_data[o]["award"] or not actors_data[d]["award"]:
            #     # print str(len(actors_data[o]["award"])) +" "+ str(len(actors_data[d]["award"]))
            #     graph.remove_edge(o, d)
        else:
            graph.remove_edge(o, d)

subgraph = max(nx.connected_component_subgraphs(graph), key=len)
# out_file = open("actor_network_weighted_overlap_cut"+str(cut)+"_awarded.csv", "w")
# out_file = open("actor_network_cut"+str(cut)+"_awarded.csv", "w")
out_file = open("actor_network_cut"+str(cut)+".csv", "w")
# nx.write_edgelist(subgraph, out_file, delimiter=",", data=('weight', 'NOverlap'))
nx.write_edgelist(subgraph, out_file, delimiter=",", data=False)
print len(subgraph.nodes())
print len(subgraph.edges())