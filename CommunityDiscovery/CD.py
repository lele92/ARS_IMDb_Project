# -*- coding: utf-8 -*-
__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import community
import matplotlib.pyplot as plt

def k_clique_CD(graph):
    # num_cliques = nx.number_of_cliques(actor_network_cut3)
    # print len(num_cliques)
    output = nx.k_clique_communities(graph,4)
    output_list = list(output)
    print 'missione completata \o/'
    for x in output_list:
        print x

def louvain_CD(graph):
    # first compute the best partition
    partition = community.best_partition(graph)

    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        print len(list_nodes)

path_actor_network = "../DATA/Network_data_final/actor_network_cleaned.csv"
path_weighted_actor_network = "../DATA/Network_data_final/actor_network_weighted.csv"
path_actor_network_cut3 = "../DATA/Network_data_final/actor_network_cut3.csv"


# actor network completa
input_actor_network = open(path_actor_network)
actor_network = nx.read_edgelist(input_actor_network, delimiter=',')

# actor network con cut 3
input_actor_network_cut3 = open(path_actor_network_cut3)
actor_network_cut3 = nx.read_edgelist(input_actor_network_cut3, delimiter=',')

# actor network con cut 4

# random network per test
# er_g = nx.erdos_renyi_graph(5500, 0.00616)


k_clique_CD(actor_network_cut3)






#drawing
# size = float(len(set(partition.values())))
# pos = nx.spring_layout(actor_network)
# count = 0.
# for com in set(partition.values()) :
#     count = count + 1.
#     list_nodes = [nodes for nodes in partition.keys()
#                                 if partition[nodes] == com]
#     print len(list_nodes)


# nx.draw_networkx_edges(actor_network,pos, alpha=0.5)
# plt.show()
