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
    f_input = open(input, "r")
    f_output = open(output, "a")
    nodes = []
    for i in f_input:
        nodes.append(i.replace('\n', ''))

    for n in nodes:
        found_communities = find_in_communities(communities,n)     # occhio agli zeri
        node_community_label = ','.join(found_communities)
        line = "%s,%s\n" % (n, node_community_label)
        f_output.write("%s" % line.encode('utf-8'))
        f_output.flush()

    f_output.close()

def find_in_communities(communities,actor):
    correct_actor = "0" * (7 - len(actor)) + actor
    found_communities = []
    for i in communities:
        if correct_actor in communities[i]:
            found_communities.append(i)
    return found_communities

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
            p, label = evaluate_single_community(community)
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

def evaluate_single_community(community_list, graph):
    community_label = None
    actors_genres = []
    right_community_items = []
    for item in community_list:
        right_item = "0"*(7-len(item))+item
        actors_genres.append(actors_data[right_item]["top_genre"])
        right_community_items.append(right_item)
        # print actors_genres
    data = Counter(actors_genres)
    community_label = data.most_common(1)[0]
    # print community_label
    purezza = float(community_label[1])/float(len(community_list))*100
    subgraph_community = graph.subgraph(right_community_items)
    density = nx.density(subgraph_community)
    return purezza, community_label[0], density

def plot_general_barchart(data, x_label, y_label, title, out, highlight=None):
    x = []
    freq = []
    for key, value in sorted(data):
        x.append(key)
        freq.append(value)
    histogram(x, freq, x_label, y_label, title, out, highlight)

def histogram(x, freq, xlabel=None, ylabel=None, title=None, out=None, highlight=None):
    # for i in range(0,len(x)-1):
    #     if (i%10 != 0):
    #         x[i] = ""

    barlist = plt.bar(range(len(freq)), freq, color='b', alpha=0.6, linewidth=0, align='center')

    if highlight is not None:
        barlist[highlight].set_color('r')
        rect = barlist[highlight]
        height = rect.get_height()
        width = rect.get_width()
        plt.text(rect.get_x() + width/2., height+1, "%.2f%%" % float(height), ha='center', va='center')

    plt.xticks(range(len(x)), x, size='x-small', rotation='vertical', ha='center', va='top')
    plt.tick_params(axis='y', labelsize='x-small')
    plt.xlim([0-barlist[0].get_width(), len(freq)])
    # plt.axis([-0.5,40.5, 0, 300])
    plt.title(title)
    plt.gca().yaxis.grid(True)

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

def evaluate_CDalgorithm_attempt_by_genre(list_communities):

    path = "../DATA/Network_data_final/actor_network_cleaned.csv"
    graph = nx.read_edgelist(path, delimiter=',', nodetype=str)

    for epsilon in list_communities:
        print epsilon
        purezza = []
        all_measure = {}
        labels = []
        total_length = 0
        count = 0
        for community in list_communities[epsilon]:
            total_length += len(community)
        for community in list_communities[epsilon]:
            p, label, density = evaluate_single_community(community, graph)
            purezza.append(p)
            labels.append(label)

            all_measure[count] = {
                "len": len(community),
                "purity": p,
                "label": label,
                "density": density
            }
            count += 1

        out_path = "KCLIQUEResult/kclique_result_"+str(epsilon)+"_k.csv"
        out = open(out_path, "w")
        for item in sorted(all_measure, key=lambda (item): (all_measure[item]["len"]), reverse=True):
            res = "%s,%s,%s,%s,%s\n" % (item, all_measure[item]["len"], all_measure[item]["purity"], all_measure[item]["density"], all_measure[item]["label"])
            out.write("%s" % res.encode('utf-8'))
            out.flush()

        out.close()

def compute_global_statistics_CDalgorithm(list_communities):

    all_ponderate_density = {}
    all_ponderate_purities = {}
    all_arithmetic_density = {}
    all_arithmetic_purities = {}
    all_unique_label = {}
    all_label = {}
    all_lenghts = {}

    for epsilon in list_communities:
        print epsilon
        labels = []
        all_lenght = []
        total_length = 0
        all_statistic_measure = list_communities[epsilon]

        ponderate_mean_purity = 0
        mean_purity = 0
        ponderate_mean_density = 0
        mean_density = 0
        for community in all_statistic_measure:
            total_length += all_statistic_measure[community]["len"]
            ponderate_mean_density += float(all_statistic_measure[community]["len"]) * float(all_statistic_measure[community]["density"])
            ponderate_mean_purity += float(all_statistic_measure[community]["len"]) * float(all_statistic_measure[community]["p"])
            mean_purity += float(all_statistic_measure[community]["p"])
            mean_density += float(all_statistic_measure[community]["density"])
            labels.append(all_statistic_measure[community]["label"])
            all_lenght.append(all_statistic_measure[community]["len"])

        mean_density /= float(total_length)
        mean_purity /= float(total_length)
        ponderate_mean_density /= float(total_length)
        ponderate_mean_purity /= float(total_length)

        all_arithmetic_density[epsilon] = mean_density
        all_ponderate_density[epsilon] = ponderate_mean_density
        all_arithmetic_purities[epsilon] = mean_purity
        all_ponderate_purities[epsilon] = ponderate_mean_purity
        all_unique_label[epsilon] = np.unique(labels).tolist()
        all_label[epsilon] = labels
        all_lenghts[epsilon] = all_lenght

    # global statistics object contains in a statistic keyword (such as arithmetic_density)
    # an object that have different object inside them, one for every epsilon (k in k-clique)
    # with the right parameter for the evaluation of the parameter.
    # For Example:
    # global_statistics {
    #   "arithmetic_density" : {
    #        0.12 : "value of arithmetic density of the community discover with epsilon 0.12 in DEMON"
    #        0.37 : "value of arithmetic density of the community discover with epsilon 0.37 in DEMON"
    #
    # for the labels we have an array of labels that represent every label of the communities discovered
    # with a certain parameter epsilon in order to plot them with an horizontal barchart
    global_statistics = {}
    global_statistics["arithmetic_density"] = all_arithmetic_density
    global_statistics["ponderate_density"] = all_ponderate_density
    global_statistics["arithmetic_purity"] = all_arithmetic_purities
    global_statistics["ponderate_purity"] = all_ponderate_purities
    global_statistics["unique_labels"] = all_unique_label
    global_statistics["labels"] = all_label
    global_statistics["lenghts"] = all_lenghts

    return global_statistics

def read_directory_result(log_directory):
    list_communities = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        list_communities[float(d.split("_")[2])] = read_single_file_result(log_directory + d)
    # contains an object for every epsilon with information about every community discover
    return list_communities

def read_single_file_result(demon_result_path):
    f = open(demon_result_path)
    all_community_measure = {}
    for l in f:
        u = l.rstrip().split(",")
        all_community_measure[u[0]] = {
            "len": float(u[1]),
            "p": float(u[2]),
            "density": float(u[3]),
            "label": u[4]
        }

    return all_community_measure

def plot_epsilon_dict(log_directory, out=None):
    l = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        l[float(d.split("_")[0])] = len(load_communities(log_directory +d))
    x = []
    freq = []
    for i in sorted(l):
        x.append(i)
        freq.append(l[i])
    histogram(x, freq, "k", "Number of communities", title="Number of Communities vs k K-CLIQUE", out=out)

def read_single_result(path):
    log_directory = "KCLIQUEResult/"
    list_communities = {}
    list_communities[float(path.split("_")[2])] = read_single_file_result(log_directory + path)
    return list_communities

def plot_communities_length_distribution(lengths, title, out):
    # lengths = []
    # for community in communities:
    #     lengths.append(len(community))

    lengths.sort()
    x = list(range(0, len(lengths)))
    plt.plot(x, lengths, "-", linewidth=2)
    plt.title(title, fontsize=15)
    plt.xlabel("CommunityID", fontsize=10, labelpad=0)
    plt.ylabel("Community Size", fontsize=10, labelpad=0)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.xlim([0, len(lengths)])
    plt.gca().yaxis.grid(True)
    plt.savefig("Plot/"+out+".jpg", bbox_inches="tight")
    # plt.legend(numpoints=1, loc=0, fontsize="x-small")
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

# list_communities = read_all_louvain_directory()
# # list_communities = read_single_demon_attempt("demon_actor_34_0.34_3.txt")
# print len(list_communities)
# all_ponderate_purities, all_unique_label = evaluate_demon_attempt_by_genre(list_communities)
# plot_general_barchart(sorted(all_ponderate_purities.iteritems(), key=lambda (k, v): v), "k", "Purity", "Purity Distribution Louvain based on Actor Genre", "purity_distribution_louvain_communities_on_genre")

# for key, value in sorted(all_unique_label.iteritems(), key=lambda (k, v): len(v)):
#     print "k: "+str(key)+" Unique Label: "+str(value)

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
# communities_list = read_all_k_clique_directory()
# evaluate_CDalgorithm_attempt_by_genre(communities_list)

# I read the result file of the execution of community discover algorithm
# list_communities_result = read_directory_result(log_directory="KCLIQUEResult/")
# now in global_statistics I have all the statistic for every parameter with I execute community discovery task (epsilon in DEMON or K in k-clique)
# global_statistics = compute_global_statistics_CDalgorithm(list_communities_result)

# plotting of the ponderate density over every epsilon
# plot_general_barchart(global_statistics["ponderate_density"].iteritems(), "K", "Density", "Ponderate Density Distribution K-CLIQUE communities", "density_distribution_k_clique_communities")

# plotting of the ponderate purity over every epsilon
# plot_general_barchart(global_statistics["ponderate_purity"].iteritems(), "k", "Purity", "Purity Distribution K-CLIQUE based on Actor Genre", "purity_distribution_k_clique_communities_on_genre") #, highlight=32

# plot the length of the different community in a specific k
k = 5.0
list_communities_result = read_single_result("kclique_result_"+str(k)+"_k.csv")
global_statistics = compute_global_statistics_CDalgorithm(list_communities_result)
plot_communities_length_distribution(global_statistics["lenghts"][k], "Distribution community size K: "+str(k), "community_size_distribution_"+str(k))

# plot_epsilon_dict(log_directory="OutputKCLIQUE/", out="number_of_community_distribution_k_clique")

# Louvain analysis
# louvain_CD(actor_network, "cut2")
# louvain_CD(actor_network_cut3, "cut3")
# louvain_CD(actor_network_cut4, "cut4")
# load_louvain_communities("cut2")
# load_louvain_communities("cut3")
# load_louvain_communities("cut4")

# add_community_label(load_communities("OutputLOUVAIN/2_louvain.txt"), "nodesWithCommunity.csv")

