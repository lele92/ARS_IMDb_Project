__author__ = 'Trappola'
# con questo file leggo i riusltati di demon e li inserisco in un dizionario cosi formattato:
# "communityLabel" : [ array contenente chiavi oggetti presenti nella community]

import os
import matplotlib.pyplot as plt
import json
import networkx as nx
import sys
from collections import Counter


import numpy as np
# np.arange(0,0.6,0.01)
# np.linspace(0,0.6,61)

def read_demon_community(demon_path):
    f = open(demon_path)
    community = {}
    for l in f:
        u = l.rstrip().split("\t")
        lista = u[1].strip("[]").split(", ")
        single_community = []
        for item in lista:
            single_community.append(item)
        # print single_community
        community[u[0]] = single_community
        # print len(community[u[0]])
    return community


def read_all_demon_directory(log_directory="OutputDemon/"):
    list_communities = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        list_communities[float(d.split("_")[3])] = list(read_demon_community(log_directory + d).values())
    return list_communities


def read_single_demon_attempt(path):
    log_directory = "OutputDemon/"
    list_communities = {}
    list_communities[float(path.split("_")[3])] = list(read_demon_community(log_directory + path).values())
    return list_communities


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


def evaluate_density_single_community(graph, community_list):
    right_community_items = []
    for item in community_list:
        right_item = "0"*(7-len(item))+item
        right_community_items.append(right_item)
    # print community_list
    # print right_community_items
    # sys.exit()
    subgraph_community = graph.subgraph(right_community_items)
    # print len(subgraph_community.edges())
    return nx.density(subgraph_community)



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


def plot_general_barchart(data, x_label, y_label, title, out):
    x = []
    freq = []
    for key, value in sorted(data):
        x.append(key)
        freq.append(value)
    histogram(x, freq, x_label, y_label, title, out)


def plot_communities_length_distribution(communities):
    lengths = []
    for community in communities:
        lengths.append(len(community))

    lengths.sort()
    x = list(range(0, len(lengths)))
    plt.plot(x, lengths, "-", markersize=5, label='<O> single point')
    plt.title("Removed Links vs Size of largest component", fontsize=15)
    plt.xlabel("Removed Links", fontsize=10, labelpad=0)
    plt.ylabel("Size of largest component", fontsize=10, labelpad=0)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    # plt.loglog()
    # plt.legend(numpoints=1, loc=0, fontsize="x-small")
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
    plt.savefig("Plot/overlap_distribution_"+str(epsilon)+".jpg", bbox_inches="tight")
    plt.show()


def plot_epsilon_dict(log_directory="OutputDemon/", out=None):
    l = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        l[float(d.split("_")[3])] = len(read_demon_community(log_directory +d))
    x = []
    freq = []
    for i in sorted(l):
        x.append(i)
        freq.append(l[i])
    histogram(x, freq, "Epsilon", "Number of communities", out)


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

    return all_ponderate_purities


def evaluate_demon_attempt_by_internal_density(list_communities):
    path = "../DATA/Network_data_final/actor_network_cleaned.csv"
    graph = nx.read_edgelist(path, delimiter=',', nodetype=str)

    #object that contains all the density (calculate with ponderate mean) for every epsilon considerate in the analysis
    all_ponderate_density = {}

    #object that contains all density (calculate with arithmetic mean) for every epsilon considerate in the analysis
    all_arithmetic_density = {}

    for epsilon in list_communities:
        print epsilon
        densities = []
        all_lenght = {}
        count = 0
        total_length = 0
        for community in list_communities[epsilon]:
            total_length += len(community)
        for community in list_communities[epsilon]:
            d = evaluate_density_single_community(graph, community)
            densities.append(d)
            all_lenght[count] = {
                "len": len(community),
                "density": d
            }
            count += 1
        mean_density = reduce(lambda x, y: x + y, densities) / len(densities)
        media_ponderata = 0
        for item in all_lenght:
            media_ponderata += float(all_lenght[item]["len"]) * float(all_lenght[item]["density"])
            # print str(all_lenght[item]["len"])+"   "+str(all_lenght[item]["density"])
        media_ponderata /= float(total_length)

        # I choose to compute the ponderate mean
        all_arithmetic_density[epsilon] = mean_density
        all_ponderate_density[epsilon] = media_ponderata

    return all_ponderate_density


# plot_epsilon_dict()
# list_communities = read_all_demon_directory()
list_communities = read_single_demon_attempt("demon_actor_34_0.34_3.txt")
# list_communities = read_single_demon_attempt("demon_actor_39_1.0_3.txt")
for epsilon in list_communities:
    plot_communities_length_distribution(list_communities[epsilon])

# path_actor = "../DATA/File_IMDb/actor_full_genre_cleaned.json"
# file_actor = open(path_actor).read()
# actors_data = json.loads(file_actor)
#
# list_communities = read_all_demon_directory()
# # list_communities = read_single_demon_attempt("demon_actor_34_0.34_3.txt")
# print len(list_communities)
# all_ponderate_purities = evaluate_demon_attempt_by_genre(list_communities)
# plot_general_barchart(sorted(all_ponderate_purities.iteritems(), key=lambda (k, v): v), "Epsilon", "Purity", "Purity Distribution DEMON based on Actor Genre", "purity_distribution_demon_communities_on_genre")

# list_communities = read_all_demon_directory()
# # list_communities = read_single_demon_attempt("demon_actor_34_0.34_3.txt")
# print len(list_communities)
# all_ponderate_densities = evaluate_density_single_community(list_communities)
# # plot_general_barchart(sorted(all_ponderate_densities.iteritems(), key=lambda (k, v): v), "Epsilon", "Density", "Ponderate Density Distribution DEMON communities", "density_distribution_demon_communities")


# for key, value in sorted(all_purezze.iteritems(), key=lambda (k, v): v):
#     print "Epsilon: "+str(key)+" Density: "+str(value)

# for key, value in sorted(all_unique_label.iteritems(), key=lambda (k, v): len(v)):
#     print "Epsilon: "+str(key)+" Unique Label: "+str(value)

# for epsilon in list_communities:
#     plot_overlap_distribution(list_communities[epsilon], epsilon)
