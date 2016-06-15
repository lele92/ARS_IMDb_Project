# -*- coding: utf-8 -*-
__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import community
import matplotlib.pyplot as plt
import os
import json
from collections import Counter

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


def k_clique_CD(graph, cut_str, k_range):
    print '########\tK-CLIQUE CD WITH K RANGE = ' + str(k_range) + '\t########'
    # num_cliques = nx.number_of_cliques(actor_network_cut3)
    # print len(num_cliques)
    for k in k_range:
        print '\n########\t'+str(k)+'-CLIQUE CD '+cut_str+' START\t########'
        output = nx.k_clique_communities(graph,k)
        output_communities_list = list(map(list, output))  # per covertire tutte le communities in liste
        print '########\t'+str(k)+'-CLIQUE CD '+cut_str+' COMPLETE\t########'
        output_file = "OutputKCLIQUE/"+str(k)+"_clique_"+cut_str+".txt"
        print '> numero di community trovate: ' + str(len(output_communities_list))
        serialize_communities(output_communities_list,output_file)

def louvain_CD(graph, cut_str):
    print '\n########\tLOUVAIN '+cut_str+' START\t########'
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

def load_k_clique_communities(cut_str, k_range):
    for k in k_range:
        print '\n########\t'+str(k)+'-CLIQUE ' + cut_str + ' ANALYSIS\t########'
        input = "OutputKCLIQUE/"+str(k)+"_clique_"+cut_str+".txt"
        k_clique_communities = load_communities(input)
        community_analysis(k_clique_communities)

def load_louvain_communities(cut_str):
    print '########\tLOUVAIN ' + cut_str + ' ANALYSIS\t########'
    input = "OutputLOUVAIN/louvain_"+cut_str+".txt"
    louvain_communities = load_communities(input)
    community_analysis(louvain_communities)

def community_analysis(communities):
    print "> numero di community: " + str(len(communities))
    count = 0
    for i in communities:
        count += len(communities[i])
    print "> numero totale di nodi: " + str(count)
    print "> dimensione media di una community: " + str(count / len(communities))
    helper = [(key, len(communities[key])) for key in communities.keys()]
    helper.sort(key=lambda x: x[1])
    print "> community più piccola: " + str(helper[0][1])
    print "> community più grande: " + str(helper[-1][1])
    print "> top 2 community:\n" + \
          str(helper[-1][0]) + " (" + str(helper[-1][1]) + " nodes)\n" + \
          str(helper[-2][0]) + " (" + str(helper[-2][1]) + " nodes)\n"
          # str(helper[-3][0]) + " (" + str(helper[-3][1]) + " nodes)\n"

def load_communities(path):
    f = open(path)
    communities= {}
    for l in f:
        u = l.rstrip().split("\t")
        nodes_list = u[1].strip("[]").split(",")
        community = []
        for node in nodes_list:
            community.append(node)
        communities[u[0]] = community
    # for i in communities:
    #     print i+"\t"+str(communities[i])
    return communities

def add_community_label(communities,output):
    input = "../DATA/Network_data_final/nodes.csv"
    f = open(input, "r")
    nodes = []
    for i in f:
        nodes.append(i)

    for n in nodes:
       found_communities = find_in_communities(communities,n)     # occhio agli zeri

def find_in_communities(communities,n):
    print communities
    for comm in communities:
        print comm #todo: finire

def read_all_k_clique_directory(log_directory="OutputKCLIQUE/"):
    list_communities = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        list_communities[float(d.split("_")[0])] = list(load_communities(log_directory + d).values())
    return list_communities


def read_all_louvain_directory(log_directory="OutputLOUVAIN/"):
    list_communities = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        list_communities[float(d.split("_")[0])] = list(load_communities(log_directory + d).values())
    return list_communities

def evaluate_demon_attempt_by_genre(list_communities):
    all_ponderate_purities = {}
    all_arithmetic_purities = {}
    all_unique_label = {}

    for epsilon in list_communities:
        print epsilon
        purezza = []
        all_lenght = {}
        labels = []
        count = 0
        total_length = 0
        for community in list_communities[epsilon]:
            total_length += len(community)
        for community in list_communities[epsilon]:
            p, label = evaluate_purity_single_community(community)
            purezza.append(p)
            labels.append(label)
            all_lenght[count] = {
                "len": len(community),
                "purity": p
            }
            count += 1
        # print min(purezza)
        # print all_lenght
        # for key, value in sorted(all_lenght.iteritems(), key=lambda (k, v): v["purity"]):
        #     print value
        mean_purezza = reduce(lambda x, y: x + y, purezza) / len(purezza)
        media_ponderata = 0
        for item in all_lenght:
            media_ponderata += float(all_lenght[item]["len"]) * float(all_lenght[item]["purity"])
        media_ponderata /= float(total_length)

        all_arithmetic_purities[epsilon] = mean_purezza
        all_ponderate_purities[epsilon] = media_ponderata
        all_unique_label[epsilon] = np.unique(labels).tolist()
        # print all_purezze

    return all_ponderate_purities, all_unique_label

def evaluate_purity_single_community(community_list):
    community_genre = None
    actors_genres = []
    for item in community_list:
        right_item = "0"*(7-len(item))+item
        actors_genres.append(actors_data[right_item]["top_genre"])
        # print actors_genres
    data = Counter(actors_genres)
    community_label = data.most_common(1)[0]
    # print community_label
    purezza = float(community_label[1])/float(len(community_list))*100
    return purezza, community_label[0]

def plot_general_barchart(data, x_label, y_label, title, out):
    x = []
    freq = []
    for key, value in sorted(data):
        x.append(key)
        freq.append(value)
    histogram(x, freq, x_label, y_label, title, out)

def histogram(x, freq, xlabel=None, ylabel=None, title=None, out=None):
    for i in range(0,len(x)-1):
        if (i%5 != 0):
            x[i] = ""

    plt.bar(range(len(freq)), freq, color='g', alpha=0.6, linewidth=0)
    plt.xticks(range(len(x)), x, size='small', rotation='vertical')
    # plt.axis([0, 41, 0, 300])
    plt.title(title)

    if (xlabel != None and ylabel != None):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if out == None:
        plt.show()
    else:
        plt.savefig("Plot/"+out+".jpg", bbox_inches="tight")
        plt.show()

def plot_overlap_distribution(communities, epsilon):
    # for i in range(0,len(x)-1):
    #     if (i%5 != 0):
    #         x[i] = ""
    actor_frequency = {}
    for community in communities:
        for a in community:
            if a in actor_frequency:
                actor_frequency[a] += 1
            else:
                actor_frequency[a] = 1
    print len(actor_frequency)
    freq = {}
    for actor in actor_frequency:
        if actor_frequency[actor] in freq:
            freq[actor_frequency[actor]] += 1
        else:
            freq[actor_frequency[actor]] = 1
    freq = sorted(freq.iteritems(), key=lambda (k, v): v, reverse=True)
    x_axis = []
    y_axis = []
    for key, value in freq:
        x_axis.append(key)
        y_axis.append(value)

    plt.bar(x_axis, y_axis, align='center', alpha=0.6, linewidth=0)
    plt.title("Overlap Distribution Epsilon: "+str(epsilon))
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.savefig("Plot/overlap_distribution_"+str(epsilon)+"_clique.jpg", bbox_inches="tight")
    plt.show()


path_actor = "../DATA/File_IMDb/actor_full_genre_cleaned.json"
file_actor = open(path_actor).read()
actors_data = json.loads(file_actor)

# list_communities = read_all_k_clique_directory()
# # list_communities = read_single_demon_attempt("demon_actor_34_0.34_3.txt")
# print len(list_communities)
# all_ponderate_purities, all_unique_label = evaluate_demon_attempt_by_genre(list_communities)
# plot_general_barchart(sorted(all_ponderate_purities.iteritems(), key=lambda (k, v): v), "k", "Purity", "Purity Distribution k-clique based on Actor Genre", "purity_distribution_k_clique_communities_on_genre")

# for key, value in sorted(all_unique_label.iteritems(), key=lambda (k, v): len(v)):
#     print "k: "+str(key)+" Unique Label: "+str(value)

# for epsilon in list_communities:
#     plot_overlap_distribution(list_communities[epsilon], epsilon)

list_communities = read_all_louvain_directory()
# # list_communities = read_single_demon_attempt("demon_actor_34_0.34_3.txt")
# print len(list_communities)
all_ponderate_purities, all_unique_label = evaluate_demon_attempt_by_genre(list_communities)
plot_general_barchart(sorted(all_ponderate_purities.iteritems(), key=lambda (k, v): v), "k", "Purity", "Purity Distribution Louvain based on Actor Genre", "purity_distribution_louvain_communities_on_genre")

for key, value in sorted(all_unique_label.iteritems(), key=lambda (k, v): len(v)):
    print "k: "+str(key)+" Unique Label: "+str(value)

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
# er_g = nx.erdos_renyi_graph(5500, 0.00616)


# K-clique analysis
k_range1 = list(range(3,11))
k_range2 = list(range(11,15))
k_range3 = list(range(15,18))
# k_clique_CD(actor_network_cut3, "cut3", k_range)       # occhio che con questa esplode tutto
# k_clique_CD(actor_network_cut4, "cut4", k_range1)
# load_k_clique_communities("cut4", k_range1)

# Louvain analysis
# louvain_CD(actor_network, "cut2")
# louvain_CD(actor_network_cut3, "cut3")
# louvain_CD(actor_network_cut4, "cut4")
# load_louvain_communities("cut2")
# load_louvain_communities("cut3")
# load_louvain_communities("cut4")

# add_community_label(load_communities("OutputLOUVAIN/2_louvain.txt"),"nodesWithCommunity.csv")

