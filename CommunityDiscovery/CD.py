# -*- coding: utf-8 -*-
__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import community
import matplotlib.pyplot as plt

def serialize_communities(comm_list, out_f):
    out_file = open(out_f, "w")
    comm_count = 0
    for comm in comm_list:
        out_file.write("%d\t[" %comm_count)
        comm_count += 1
        for n in comm[:-1]:
            out_file.write("%s," %n)
        out_file.write("%s]\n" %comm[-1])           # barbatrucco per non avere la virgola alla fine della lista
    out_file.close()


def k_clique_CD(graph, cut_str):
    k_range = list(range(2,15))
    # num_cliques = nx.number_of_cliques(actor_network_cut3)
    # print len(num_cliques)
    print '########\tK-CLIQUE '+cut_str+' START\t########'
    output = nx.k_clique_communities(graph,4)
    output_communities_list = list(map(list, output))  # per covertire tutte le communities in liste
    print '########\tK-CLIQUE '+cut_str+' COMPLETE\t########'
    output_file = "OutputKCLIQUE/4_clique_"+cut_str+".txt"
    print '> numero di community trovate: ' + str(len(output_communities_list))
    serialize_communities(output_communities_list,output_file)

def louvain_CD(graph, cut_str):
    print '########\tLOUVAIN '+cut_str+' START\t########'
    # first compute the best partition
    partition = community.best_partition(graph)
    print '########\tLOUVAIN '+cut_str+' COMPLETE\t########'
    output_file = "OutputLOUVAIN/louvain_"+cut_str+".txt"
    comm_list = []
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        comm_list.append(list_nodes)
    print '> numero di community trovate: '+str(len(comm_list))
    serialize_communities(comm_list, output_file)

path_actor_network = "../DATA/Network_data_final/actor_network_cleaned.csv"
path_weighted_actor_network = "../DATA/Network_data_final/actor_network_weighted.csv"
path_actor_network_cut4 = "../DATA/Network_data_final/actor_network_cut4.csv"
path_actor_network_cut3 = "../DATA/Network_data_final/actor_network_cut3.csv"


# actor network completa (cut 2)
input_actor_network = open(path_actor_network)
actor_network = nx.read_edgelist(input_actor_network, delimiter=',')

# actor network con cut 3
input_actor_network_cut3 = open(path_actor_network_cut3)
actor_network_cut3 = nx.read_edgelist(input_actor_network_cut3, delimiter=',')

# actor network con cut 4
input_actor_network_cut4 = open(path_actor_network_cut4)
actor_network_cut4 = nx.read_edgelist(input_actor_network_cut4, delimiter=',')


# random network per test
er_g = nx.erdos_renyi_graph(5500, 0.00616)

# K-clique analysis
# k_clique_CD(actor_network_cut3, "cut3")       # occhio che con questa esplode tutto
k_clique_CD(actor_network_cut4, "cut4")


# Louvain analysis
louvain_CD(actor_network, "cut2")
louvain_CD(actor_network_cut3, "cut3")
louvain_CD(actor_network_cut4, "cut4")
