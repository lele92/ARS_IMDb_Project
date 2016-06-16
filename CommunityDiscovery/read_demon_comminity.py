__author__ = 'Trappola'
# con questo file leggo i riusltati di demon e li inserisco in un dizionario cosi formattato:
# "communityLabel" : [ array contenente chiavi oggetti presenti nella community]

import os
import matplotlib.pyplot as plt
import json
import networkx as nx
import sys
from collections import Counter

min_community_size = 3

import numpy as np
# np.arange(0,0.6,0.01)
# np.linspace(0,0.6,61)


def read_single_file_demon_result(demon_result_path):
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


def read_all_demon_directory_result(log_directory="DEMONResult/"):
    list_communities = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        list_communities[float(d.split("_")[2])] = read_single_file_demon_result(log_directory + d)
    # contains an object for every epsilon with information about every community discover
    return list_communities


def read_single_demon_attempt(path):
    log_directory = "OutputDemon/"
    list_communities = {}
    list_communities[float(path.split("_")[3])] = list(read_demon_community(log_directory + path).values())
    return list_communities


def read_single_demon_result(path):
    log_directory = "DEMONResult/"
    list_communities = {}
    list_communities[float(path.split("_")[2])] = read_single_file_demon_result(log_directory + path)
    return list_communities


def evaluate_purity_single_community(community_list, graph):
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


def horizontal_barchar(data):
    freq = {}
    tot_item = len(data)
    for item in data:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    freq = sorted(freq.iteritems(), key=lambda (k, v): v, reverse=True)

    y_axis = []
    labels = []
    max_width = sorted(freq, key=lambda (k, v): v, reverse=True)[0][1]/float(tot_item)*100
    for key, value in sorted(freq, key=lambda (k, v): v, reverse=True):
        labels.append(key)
        y_axis.append(float(value)/float(tot_item)*100)


    rects = plt.barh(range(len(freq)), y_axis, color='#0000CC', alpha=0.8, linewidth=0, align='center')
    bar_width = 0
    for rect in rects:
        height = rect.get_height()
        width = rect.get_width()
        bar_width = height
        width = "%.2f" % width
        plt.text(5+float(width), rect.get_y() + height/2., "%.1f%%" % float(width), ha='center', va='center', fontsize=7)
    plt.axis([0, max_width + 10, -0.5, len(labels)-0.5])
    plt.yticks(np.arange(len(labels)), labels, ha='right', va='center', size='small', rotation='horizontal')
    plt.show()


def histogram(x, freq, xlabel=None, ylabel=None, title=None, out=None, highlight=None):
    for i in range(0,len(x)-1):
        if (i%10 != 0):
            x[i] = ""

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


def plot_general_barchart(data, x_label, y_label, title, out, highlight=None):
    x = []
    freq = []
    for key, value in sorted(data):
        x.append(key)
        freq.append(value)
    histogram(x, freq, x_label, y_label, title, out, highlight)


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
    histogram(x, freq, "Epsilon", "Number of communities", title="Number of Communities vs Epsilon DEMON", out=out)


def evaluate_demon_attempt_by_genre(list_communities):

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
            p, label, density = evaluate_purity_single_community(community, graph)
            purezza.append(p)
            labels.append(label)
            all_measure[count] = {
                "len": len(community),
                "purity": p,
                "label": label,
                "density": density
            }
            count += 1

        out_path = "DEMONResult/demon_result_"+str(epsilon)+"_"+str(min_community_size)+".csv"
        out = open(out_path, "w")
        for item in sorted(all_measure, key=lambda (item): (all_measure[item]["len"]), reverse=True):
            res = "%s,%s,%s,%s,%s\n" % (item, all_measure[item]["len"], all_measure[item]["purity"], all_measure[item]["density"], all_measure[item]["label"])
            out.write("%s" % res.encode('utf-8'))
            out.flush()

        out.close()


def compute_global_statistics_DEMON_attempt(list_communities):

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


# plot_epsilon_dict(out="number_of_community_distribution")
# list_communities = read_all_demon_directory()
epsilon = 0.32
list_communities_result = read_single_demon_result("demon_result_0.32_3.csv")
# list_communities = read_single_demon_attempt("demon_actor_39_1.0_3.txt")
# for epsilon in list_communities:
#     plot_communities_length_distribution(list_communities[epsilon], "Distribution community size Epsilon: "+str(epsilon), "community_size_distribution_"+str(epsilon))

# path_actor = "../DATA/File_IMDb/actor_full_genre_cleaned.json"
# file_actor = open(path_actor).read()
# actors_data = json.loads(file_actor)

# list_communities = read_all_demon_directory()
# epsilon = 0.32
# list_communities = read_single_demon_attempt("demon_actor_32_"+str(epsilon)+"_3.txt")
# print len(list_communities)
# evaluate_demon_attempt_by_genre(list_communities)
# horizontal_barchar(labels[epsilon])
# plot_general_barchart(sorted(all_ponderate_purities.iteritems(), key=lambda (k, v): v), "Epsilon", "Purity", "Purity Distribution DEMON based on Actor Genre", "purity_distribution_demon_communities_on_genre", highlight=32)

# # Plot overlap distribution of every attempt of different parameter in community discovery
# for epsilon in list_communities:
#     plot_overlap_distribution(list_communities[epsilon], epsilon)

# # I read the result file of the execution of community discover algorithm
# list_communities_result = read_all_demon_directory_result()
# # now in global_statistics I have all the statistic for every parameter with I execute community discovery task (epsilon in DEMON or K in k-clique)
# global_statistics = compute_global_statistics_DEMON_attempt(list_communities_result)

# # print on console the list of different label that describe the different communities found with a certain epsilon
# for key, value in sorted(global_statistics["unique_labels"].iteritems(), key=lambda (k, v): len(v)):
#     print "Epsilon: "+str(key)+" Unique Label: "+str(value)

# # plotting of the ponderate density over every epsilon
# plot_general_barchart(sorted(global_statistics["ponderate_density"].iteritems(), key=lambda (k, v): v), "Epsilon", "Density", "Ponderate Density Distribution DEMON communities", "density_distribution_demon_communities")

# # plotting of the ponderate purity over every epsilon
# plot_general_barchart(sorted(global_statistics["ponderate_purity"].iteritems(), key=lambda (k, v): v), "Epsilon", "Purity", "Purity Distribution DEMON based on Actor Genre", "purity_distribution_demon_communities_on_genre", highlight=32)

# # plot the length of the different community in a specific epsilon
# for epsilon in global_statistics["lenghts"]:
#     plot_communities_length_distribution(global_statistics["lenghts"][epsilon], "Distribution community size Epsilon: "+str(epsilon), "community_size_distribution_"+str(epsilon))
