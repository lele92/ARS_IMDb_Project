# -*- coding: utf-8 -*-
__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

import json
import networkx as nx

candidates_ids_file = open("oscar_candidate_id.txt")
for l in candidates_ids_file:
    candidates_ids = eval(l)

GRAPH_PATH = "../DATA/Network_data_final/actor_network_cut3_awarded.csv"
graph = nx.read_edgelist(GRAPH_PATH, delimiter=',', nodetype=str)

GRAPH_PATH = "../DATA/Network_data_final/actor_network_cleaned.csv"
graph_plain = nx.read_edgelist(GRAPH_PATH, delimiter=',', nodetype=str)

PATH_ACTOR = "../DATA/File_IMDb/actor_full_genre_cleaned.json"
file_actor = open(PATH_ACTOR).read()
actors_data = json.loads(file_actor)

out = open("neighbor_distribution_candidate.txt", "w")

# Andiamo a vedere la distribuzione dei vicini degli attori nella lista dei candidati
# all'interno della rete con cut a 3 di soli attori nominati
for candidate in candidates_ids:
    neighbors = graph.neighbors(candidate)
    neighbors_types = {
        "Oscar": 0,
        "NominationOscar": 0,
        "Winning": 0,
        "GenericNomination": 0
    }
    from_leo = nx.shortest_path_length(graph, candidate, "0000138")
    from_leo_plain = nx.shortest_path_length(graph_plain, candidate, "0000138")

    for actor in neighbors:
        neighbors_types[actors_data[actor]["award_category"]] += 1

    res = "%s,%s,%s,%s,%s,%s\n" % (candidate, actors_data[candidate]["name"], len(neighbors), neighbors_types, from_leo,from_leo_plain)
    out.write("%s" % res.encode('utf-8'))
    out.flush()
out.close()


