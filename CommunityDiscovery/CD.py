# -*- coding: utf-8 -*-
__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

import networkx as nx
import numpy as np
import community
import matplotlib.pyplot as plt
import os
import json
from collections import Counter
from Demon import Demon

DEMON = "demon"
KCLIQUE = "kclique"
LOUVAIN = "louvain"

MIN_COMMUNITY_SIZE_DEMON = 3
RESULT_DIRECTORY_DEMON_AWARD = "Demon_Award_Result"
RESULT_DIRECTORY_DEMON = "DEMONResults"
RESULT_DIRECTORY_LOUVAIN = "LOUVAINResult"
RESULT_DIRECTORY_KCLIQUE = "KCLIQUEResult"

OUTPUT_DIRECTORY_DEMON_AWARD = "OutputDemon_Award"
OUTPUT_DIRECTORY_KCLIQUE_AWARD = "OutputKCLIQUE_Award"
OUTPUT_DIRECTORY_LOUVAIN_AWARD = "OutputLOUVAIN_Award"
OUTPUT_DIRECTORY_DEMON = "OutputDemon"
OUTPUT_DIRECTORY_KCLIQUE = "OutputKCLIQUE"
OUTPUT_DIRECTORY_LOUVAIN = "OutputLOUVAIN"


GRAPH_PATH_AWARDED = "../DATA/Network_data_final/actor_network_cut3_awarded.csv"
GRAPH_PATH = "../DATA/Network_data_final/actor_network_cleaned.csv"

PLOT_DIRECTORY_AWARD = "PlotAward"
PLOT_DIRECTORY_DEMON = "Plot_DEMON"
PLOT_DIRECTORY_KCLIQUE = "Plot_KCLIQUE"
PLOT_DIRECTORY_LOUVAIN = "Plot_LOUVAIN"
PATH_ACTOR = "../DATA/File_IMDb/actor_full_genre_cleaned.json"

path_actor_network = "../DATA/Network_data_final/actor_network_cleaned.csv"
path_weighted_actor_network = "../DATA/Network_data_final/actor_network_weighted.csv"
path_actor_network_cut4 = "../DATA/Network_data_final/actor_network_cut4.csv"
path_actor_network_cut3 = "../DATA/Network_data_final/actor_network_cut3.csv"
path_actor_network_cut3_awarded = "../DATA/Network_data_final/actor_network_cut3_awarded.csv"

OSCAR_TRESHOLD = 20


# execution of DEMON algorithm for CD task
def find_communities_demon(network_path, epsilon_start, delta, attempt, min_community_size, output_dir):
    attempt_range = range(0, attempt)
    epsilon = epsilon_start
    for i in attempt_range:
        file_output = output_dir+"/demon_actor_"+str(epsilon)+"_"+str(min_community_size)
        dm = Demon(network_path, epsilon=epsilon, min_community_size=min_community_size, file_output=file_output)
        dm.execute()
        epsilon += delta


# execution of KCLIQUE algorithm for CD task
def k_clique_CD(graph, cut_str, k_range):
    print '########\tK-CLIQUE CD WITH K RANGE = ' + str(k_range) + '\t########'
    # num_cliques = nx.number_of_cliques(actor_network_cut3)
    # print len(num_cliques)
    for k in k_range:
        print '\n########\t'+str(k)+'-CLIQUE CD '+cut_str+' START\t########'
        output = nx.k_clique_communities(graph, k)
        output_communities_list = list(map(list, output))  # per covertire tutte le communities in liste
        print '########\t'+str(k)+'-CLIQUE CD '+cut_str+' COMPLETE\t########'
        output_file = OUTPUT_DIRECTORY_KCLIQUE+"/kclique_actor_"+str(k)+"_"+cut_str+".txt"
        print '> numero di community trovate: ' + str(len(output_communities_list))
        serialize_communities(output_communities_list, output_file)


# execution of LUOVAIN algorithm for CD task
def louvain_CD(graph, cut_str):
    print '\n########\tLOUVAIN '+cut_str+' START\t########'
    # first compute the best partition
    partition = community.best_partition(graph,weight='weight',)
    print '########\tLOUVAIN '+cut_str+' COMPLETE\t########'
    output_file = OUTPUT_DIRECTORY_LOUVAIN+"/louvain_actor_"+cut_str+"_.txt"
    comm_list = []
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        comm_list.append(list_nodes)
    print '> numero di community trovate: '+str(len(comm_list))
    serialize_communities(comm_list, output_file)


# result serialization CD algorithm execution (DEMON like)
def serialize_communities(comm_list, out_f):
    out_file = open(out_f, "w")
    comm_count = 0
    for comm in comm_list:
        out_file.write("%d\t[" %comm_count)
        comm_count += 1
        for n in comm[:-1]:
            out_file.write("%s, " %n)
        out_file.write("%s]\n" %comm[-1])
    out_file.close()


def k_clique_communities_basic_analysis(cut_str, k_range):
    for k in k_range:
        print '\n########\t'+str(k)+'-CLIQUE ' + cut_str + ' ANALYSIS\t########'
        input = "OutputKCLIQUE/kclique_actor_"+cut_str+".txt"
        k_clique_communities = load_communities(input)

        community_analysis(k_clique_communities)


def louvain_communities_basic_analysis(cut_str):
    print '########\tLOUVAIN ' + cut_str + ' ANALYSIS\t########'
    input = "OutputLOUVAIN/louvain_actor_"+cut_str+"_.txt"
    louvain_communities = load_communities(input)
    community_analysis(louvain_communities)


# basic statistics of a single CD algorith execution result
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


# load communities of a single CD algorith execution result
def load_communities(path):
    f = open(path)
    communities = {}
    for l in f:
        u = l.rstrip().split("\t")
        nodes_list = u[1].strip("[]").split(", ")
        community = []
        for node in nodes_list:
            community.append(node)
        communities[u[0]] = community
    # for i in communities:
    #     print i+"\t"+str(communities[i])
    return communities


# add community label at all the node of the network in order to visualize CD result with Gephi
def add_community_label(communities, output):
    input = "../DATA/Network_data_final/nodes.csv"
    f_input = open(input, "r")
    f_output = open(output, "a")
    nodes = []
    for i in f_input:
        nodes.append(i.replace('\n', ''))

    for n in nodes:
        found_communities = find_in_communities(communities, n)
        node_community_label = ','.join(found_communities)
        n_str = str("0" * (7 - len(n)) + n)
        line = "%s,%s\n" % (n_str, node_community_label)
        f_output.write("%s" % line.encode('utf-8'))
        f_output.flush()

    f_output.close()


    # add community label at all the node of the network in order to visualize CD result with Gephi
def add_community_label_from_graph(communities, output, graph):
    f_output = open(output, "a")

    for n in graph.nodes():
        found_communities = find_in_communities_borghi(communities, n)
        node_community_label = ','.join(found_communities)
        line = "%s,%s\n" % (n, node_community_label)
        f_output.write("%s" % line.encode('utf-8'))
        f_output.flush()

    f_output.close()


# add community label at all the node of the network in order to visualize CD result with Gephi
def add_award_category_label(output):

    f_output = open(output, "a")

    for actor in actor_network_cut3_awarded.nodes():
        line = "%s,%s\n" % (actor, len(actors_data[actor]["award_category"]))
        f_output.write("%s" % line.encode('utf-8'))
        f_output.flush()

    f_output.close()


# find the community of an actor
def find_in_communities(communities, actor):
    correct_actor = "0" * (7 - len(actor)) + actor
    found_communities = []
    for i in communities:
        if correct_actor in communities[i]:
            found_communities.append(i)
    return found_communities


# find the community of an actor
def find_in_communities_borghi(communities, node):
    # correct_actor = "0" * (7 - len(actor)) + actor
    found_communities = []
    for i in communities:
        if node in communities[i]:
            found_communities.append(i)
    return found_communities


def read_single_CD_algorithm_attempt(log_directory, path):
    list_communities = {}
    list_communities[float(path.split("_")[2])] = load_communities(log_directory +"/"+ path)
    return list_communities


def read_all_CD_algorithm_output_directory(log_directory):
    list_communities = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        list_communities[float(d.split("_")[2])] = load_communities(log_directory +"/"+ d)
    return list_communities


# evaluate all the community found (for different input parameter) with different kind of evaluation,
# internal with density and external with purity
def evaluate_CD_algorithm_attempt(list_communities, result_directory, algorithm_apply, min_community_size=""):

    graph = nx.read_edgelist(GRAPH_PATH_AWARDED, delimiter=',', nodetype=str)
    oscar_community_eps = {}

    for parameter in list_communities:
        print parameter
        purezza = []
        all_measure = {}
        labels = []
        total_length = 0
        all_com_eps = list_communities[parameter]
        oscar_community = {}

        for community in all_com_eps:
            total_length += len(all_com_eps[community])

        for community in all_com_eps:
            p, label, density, mean_birth_date, percentage_birth, oscar_percentage = evaluate_single_community(all_com_eps[community], graph)
            purezza.append(p)
            labels.append(label)
            all_measure[community] = {
                "len": len(all_com_eps[community]),
                "purity": p,
                "label": label,
                "density": density,
                "birth_date": mean_birth_date,
                "percentage_birth": percentage_birth
            }

            if oscar_percentage and oscar_percentage >= OSCAR_TRESHOLD:
                oscar_community[community] = all_com_eps[community]

        out_path = result_directory+"/"+algorithm_apply+"_result_"+str(parameter)+"_"+str(min_community_size)+".csv"
        out = open(out_path, "w")
        for item in sorted(all_measure, key=lambda (item): (all_measure[item]["len"]), reverse=True):
            res = "%s,%s,%s,%s,%s,%s,%s\n" % (item, all_measure[item]["len"], all_measure[item]["purity"], all_measure[item]["density"], all_measure[item]["label"], all_measure[item]["birth_date"], all_measure[item]["percentage_birth"])
            out.write("%s" % res.encode('utf-8'))
            out.flush()

        out.close()
        oscar_community_eps[parameter] = oscar_community

    return oscar_community_eps


# evaluate single community found with different kind of evaluation,
# internal with density and external with purity
def evaluate_single_community(community_list, graph):
    # print community_list
    community_label = None
    actors_genres = []
    right_community_items = []
    birth_date = []
    actors_years = []
    for item in community_list:
        right_item = "0"*(7-len(item))+item
        actors_genres.append(actors_data[right_item]["top_genre"])
        # if "award_category" in actors_data[right_item]:
        #     actors_genres.append(actors_data[right_item]["award_category"])
        right_community_items.append(right_item)
        actors_years.append(actors_data[right_item]["mean_year_film"])
        if actors_data[right_item]["birth date"] is not None:
            birth_date.append(int(actors_data[right_item]["birth date"]))

    data = Counter(actors_genres)
    community_label = data.most_common(1)[0]
    purezza = float(community_label[1])/float(len(community_list))*100
    # purezza = -1
    subgraph_community = graph.subgraph(right_community_items)
    density = nx.density(subgraph_community)

    mean_birth_date = 0
    percentage_birth_date = 0.00
    if birth_date:
        mean_birth_date = reduce(lambda x, y: x + y, birth_date) / len(birth_date)
        percentage_birth_date = float(len(birth_date))/float(len(community_list))*100
    # mean_year_actor = reduce(lambda x, y: x + y, actors_years) / len(actors_years)

    oscar_percentage = None
    if "Oscar" in data:
        # purezza = data["Oscar"]
        oscar_percentage = float(data["Oscar"])/float(len(community_list))*100

    return purezza, community_label[0], density, mean_birth_date, "%.2f" % percentage_birth_date, oscar_percentage
    # return purezza, data, density, mean_birth_date, "%.2f" % percentage_birth_date, mean_year_actor


def plot_general_barchart(data, plot_directory, x_label, y_label, title, out, highlight=None):
    x = []
    freq = []
    for key, value in data:
        x.append(key)
        freq.append(value)
    histogram(x, freq, plot_directory, x_label, y_label, title, out, highlight)


def histogram(x, freq, plot_directory, xlabel=None, ylabel=None, title=None, out=None, highlight=None):
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
    # plt.title(title)
    plt.gca().yaxis.grid(True)

    if (xlabel != None and ylabel != None):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if out == None:
        plt.show()
    else:
        plt.savefig(plot_directory+"/"+out+".jpg", bbox_inches="tight")
        plt.show()


def plot_overlap_distribution(communities, plot_directory, CD_param):
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
    # print len(actor_frequency)

    freq = {}
    for actor in actor_frequency:
        if actor_frequency[actor] in freq:
            freq[actor_frequency[actor]] += 1
        else:
            freq[actor_frequency[actor]] = 1
    freq = sorted(freq.iteritems(), key=lambda (k, v): v, reverse=True)
    # print freq
    x_axis = []
    y_axis = []
    for key, value in freq:
        x_axis.append(key)
        y_axis.append(value)

    plt.bar(x_axis, y_axis, align='center', alpha=0.6, linewidth=0)
    # plt.title("Overlap Distribution k: " + str(CD_param))
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.savefig(plot_directory+"/overlap_distribution_" + str(CD_param) + "_clique.jpg", bbox_inches="tight")
    plt.show()


def success_communities_analysis(success_community):
    actor_frequency = {}
    for community in success_community:
        for a in success_community[community]:
            if a in actor_frequency:

                actor_frequency[a] += 1
            else:
                actor_frequency[a] = 1

    # print len(actor_frequency)
    freq = {}
    count_oscar = 0

    candidates_ids_file = open("oscar_candidate_id.txt")
    for l in candidates_ids_file:
        candidates_ids = eval(l)

    for actor in actor_frequency:
        right_item = "0"*(7-len(actor))+actor
        if actors_data[right_item]["award_category"] == "Oscar":
            count_oscar += 1

        if right_item in candidates_ids:
            print actor +" "+str(actor_frequency[actor])
        # print str(right_item) +" "+str(actor_frequency[actor])+" "+actors_data[right_item]["award_category"]+" "+actors_data[right_item]["birth date"]+" "+str(len(actors_data[right_item]["award"]))
        if actor_frequency[actor] in freq:
            freq[actor_frequency[actor]] += 1
        else:
            freq[actor_frequency[actor]] = 1

    print "Count Oscar: "+str(count_oscar)


def compute_global_statistics_CD_algorithm(list_communities):

    all_ponderate_density = {}
    all_ponderate_purities = {}
    all_arithmetic_density = {}
    all_arithmetic_purities = {}
    all_unique_label = {}
    all_label = {}
    all_lenghts = {}
    all_mean_lenghts = {}

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
        num_community = len(all_statistic_measure)

        for community in all_statistic_measure:
            total_length += all_statistic_measure[community]["len"]
            ponderate_mean_density += float(all_statistic_measure[community]["len"]) * float(all_statistic_measure[community]["density"])
            ponderate_mean_purity += float(all_statistic_measure[community]["len"]) * float(all_statistic_measure[community]["p"])
            mean_purity += float(all_statistic_measure[community]["p"])
            mean_density += float(all_statistic_measure[community]["density"])
            labels.append(all_statistic_measure[community]["label"])
            all_lenght.append(all_statistic_measure[community]["len"])

        mean_density /= float(num_community)
        mean_purity /= float(num_community)
        ponderate_mean_density /= float(total_length)
        ponderate_mean_purity /= float(total_length)

        all_arithmetic_density[epsilon] = mean_density
        all_ponderate_density[epsilon] = ponderate_mean_density
        all_arithmetic_purities[epsilon] = mean_purity
        all_ponderate_purities[epsilon] = ponderate_mean_purity
        all_unique_label[epsilon] = np.unique(labels).tolist()
        all_label[epsilon] = labels
        all_lenghts[epsilon] = all_lenght
        all_mean_lenghts[epsilon] = reduce(lambda x, y: x + y, all_lenght) / len(all_lenght)
        # print all_mean_lenghts[epsilon]

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
    global_statistics["mean_lenght"] = all_mean_lenghts

    return global_statistics


def read_CD_algorithm_result_directory(log_directory):
    list_communities = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        list_communities[float(d.split("_")[2])] = read_single_file_result(log_directory +"/"+ d)
    # contains an object for every parameter with information about every community discover
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


def number_of_communities_plot(log_directory, label_y_axis, out=None):
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
    histogram(x, freq, label_y_axis, "Number of communities", title="", out=out)


def read_single_CD_Algorithm_result(log_directory, path):
    list_communities = {}
    list_communities[float(path.split("_")[2])] = read_single_file_result(log_directory +"/"+ path)
    return list_communities


def plot_communities_length_distribution(lengths, title, plot_directory, out):
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
    plt.savefig(plot_directory+"/"+out+".jpg", bbox_inches="tight")
    # plt.legend(numpoints=1, loc=0, fontsize="x-small")
    plt.show()


def horizontal_barchar(data, plot_directory, out):
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
    # max_width = sorted(freq, key=lambda (k, v): v, reverse=True)[0][1]/float(tot_item)*100
    max_width = sorted(freq, key=lambda (k, v): v, reverse=True)[0][1]
    for key, value in sorted(freq, key=lambda (k, v): v, reverse=True):
        labels.append(key)
        y_axis.append(value)
        # y_axis.append(float(value)/float(tot_item)*100)

    rects = plt.barh(range(len(freq)), y_axis, color='#0000CC', alpha=0.8, linewidth=0, align='center')
    for rect in rects:
        height = rect.get_height()
        width = rect.get_width()
        # width = "%.2f" % width
        percent_width = "%.2f" % (float(width)/float(tot_item)*100)
        plt.text(5+float(width), rect.get_y() + height/2., str(int(width))+ " (" + str(percent_width) + "%)", ha='center', va='center', fontsize=11)

    # plt.axis([0, 10 + 10, -0.5, len(labels)-0.5])
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.axis([0, max_width + 10, -0.5, len(labels)-0.5])
    plt.yticks(np.arange(len(labels)), labels, ha='right', va='center', size='small', rotation='horizontal')
    plt.savefig(plot_directory+"/"+out+".jpg", bbox_inches="tight")
    plt.show()


def compute_modularity(partiton, graph):
    m = community.modularity(partiton, graph)
    print '> modularity:' + str(m)


def get_communities_measure(communities, measure):
    communities_measure = {}
    for c in communities:
        communities_measure[c] = communities[c][measure]
        # print c + " "+str(measure)+": " + str(communities[c][measure])
    communities_measure = sorted(communities_measure.iteritems(), key=lambda (k, v): v)
    print communities_measure
    return communities_measure


file_actor = open(PATH_ACTOR).read()
actors_data = json.loads(file_actor)

# # actor network completa (cut 2)
# input_actor_network = open(path_actor_network)
# actor_network = nx.read_edgelist(input_actor_network, delimiter=',')
#
# # actor network con cut 3
# input_actor_network_cut3 = open(path_actor_network_cut3)
# actor_network_cut3 = nx.read_edgelist(input_actor_network_cut3, delimiter=',')
#
# actor network con cut 4
input_actor_network_cut4 = open(path_actor_network_cut4)
actor_network_cut4 = nx.read_edgelist(input_actor_network_cut4, delimiter=',')
#
# actor network weighted
input_actor_network_weighted = open(path_weighted_actor_network)
actor_network_weighted = nx.read_edgelist(input_actor_network_weighted, delimiter=',', data=(('weight', float),))
#
# actor network cut 3 awarded
input_actor_network_cut_3_awarded = open(path_actor_network_cut3_awarded)
actor_network_cut3_awarded = nx.read_edgelist(input_actor_network_cut_3_awarded, delimiter=',')

# random network per test
# er_g = nx.erdos_renyi_graph(5500, 0.00616)

####################################### K-CLIQUE analysis ############################################################

# k_range1 = list(range(3, 13))
k_range1 = [4]
# k_clique_CD(actor_network_cut3_awarded, "cut3", k_range1)       # occhio che con questa esplode tutto

# k_clique_CD(actor_network_cut4, "cut4", k_range1)
# k_clique_communities_basic_analysis("4_cut4", k_range1)
#
# communities_list = read_all_CD_algorithm_output_directory(log_directory=OUTPUT_DIRECTORY_KCLIQUE)
# evaluate_CD_algorithm_attempt(communities_list, RESULT_DIRECTORY_KCLIQUE, KCLIQUE)

# I read the result file of the execution of community discover algorithm
# list_communities_result = read_CD_algorithm_result_directory(log_directory=RESULT_DIRECTORY_KCLIQUE)
# now in global_statistics I have all the statistic for every parameter with I execute community discovery task (epsilon in DEMON or K in k-clique)
# global_statistics = compute_global_statistics_CD_algorithm(list_communities_result)

# plotting of the ponderate density over every k
# plot_general_barchart(global_statistics["arithmetic_density"].iteritems(), PLOT_DIRECTORY_KCLIQUE, "K", "Density", "Density Distribution K-CLIQUE communities", "density_distribution_k_clique_communities")

# plotting of the ponderate purity over every k
# plot_general_barchart(global_statistics["ponderate_purity"].iteritems(), PLOT_DIRECTORY_KCLIQUE, "k", "Purity", "Purity Distribution K-CLIQUE based on Actor Genre", "purity_distribution_k_clique_communities_on_genre") #, highlight=32

# plot the length of the different community in a specific k
# for k in [3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]:
#     list_communities_result = read_single_CD_Algorithm_result(RESULT_DIRECTORY_KCLIQUE ,"kclique_result_"+str(k)+"_k.csv")
#     plot_communities_length_distribution(global_statistics["lenghts"][k], "", PLOT_DIRECTORY_KCLIQUE, "community_size_distribution_"+str(k))

# list_communities_result = read_single_CD_Algorithm_result(RESULT_DIRECTORY_KCLIQUE,"kclique_result_"+str(4.0)+"_k.csv")
# sorted_comm_len = get_communities_measure(list_communities_result[4.0], 'len')
# plot_general_barchart(sorted_comm_len, PLOT_DIRECTORY_KCLIQUE, "Community ID", "length", "", "kclique_communities_lengths_distribution")

# number_of_communities_plot(log_directory=OUTPUT_DIRECTORY_KCLIQUE, out="number_of_community_distribution_k_clique", label_y_axis="k")

# print "k = " + str(k) + " unique labels = " + str(global_statistics["unique_labels"][k])
# horizontal_barchar(global_statistics["labels"][4],PLOT_DIRECTORY_KCLIQUE, "genre_distr_4")


######################################### LOUVAIN analysis #########################################################

CAST_file = "network_REdmo_10_cleaned.csv"
CAST_network = nx.read_edgelist(CAST_file, delimiter=',', data=(('weight', float),))
# louvain_CD(CAST_network, "777")
add_community_label_from_graph(load_communities("OutputLOUVAIN/louvain_actor_777_.txt"), "nodesDMOwithCommunity.csv", CAST_network)


# il parametro 2.0 sta ad indicare che tipo di rete è stato utilizzato ( 2 = cut2)
# louvain_CD(actor_network_weighted, "2.0")

# louvain_communities_basic_analysis("2.0")
# compute_modularity(community.best_partition(actor_network_weighted), actor_network_weighted)
#
# communities_list = read_all_CD_algorithm_output_directory(log_directory=OUTPUT_DIRECTORY_LOUVAIN)
# oscar_community = evaluate_CD_algorithm_attempt(communities_list,RESULT_DIRECTORY_LOUVAIN, LOUVAIN)
# list_communities_result = read_CD_algorithm_result_directory(log_directory=RESULT_DIRECTORY_LOUVAIN)

# basic analysis, modularity, global statistics e unique labels
# louvain_communities_basic_analysis("2")
# compute_modularity(community.best_partition(actor_network_weighted),actor_network_weighted)
# global_statistics = compute_global_statistics_CD_algorithm(list_communities_result)

# print "LOUVAIN communities unique labels = " + str(global_statistics["unique_labels"][2.0])

# plot density
# plot_general_barchart(global_statistics["arithmetic_density"].iteritems(), PLOT_DIRECTORY_LOUVAIN, "", "Density", "Density Distribution LOUVAIN communities", "density_distribution_LOUVAIN_communities")


# plot purity e global ponderate purity
# plot_general_barchart(global_statistics["ponderate_purity"].iteritems(), PLOT_DIRECTORY_LOUVAIN, "", "Purity", "Purity Distribution LOUVAIN based on Actor Genre", "purity_distribution_LOUVAIN_communities_on_genre") #, highlight=32
# sorted_comm_purity = get_communities_measure(list_communities_result[2.0], 'p')
# plot_general_barchart(sorted_comm_purity, PLOT_DIRECTORY_LOUVAIN, "Community ID", "Purity", "", "louvain_communities_purity_distribution")
# print "purezza media ponderata: "+str(global_statistics["ponderate_purity"][2.0])

# list_communities_result = read_single_CD_Algorithm_result(RESULT_DIRECTORY_LOUVAIN, "LOUVAIN_result_2.0_cut.csv")
# print list_communities_result

# print communities lengths distribution
# sorted_comm_len = get_communities_measure(list_communities_result[2.0], 'len')
# plot_general_barchart(sorted_comm_len, PLOT_DIRECTORY_LOUVAIN, "Community ID", "length", "", "louvain_communities_lengths_distribution")
# plot_communities_length_distribution(global_statistics["lenghts"][2.0], "", PLOT_DIRECTORY_LOUVAIN, "community_size_distribution_"+str(2.0))


# plot genre distribution
# horizontal_barchar(global_statistics["labels"][2.0] , PLOT_DIRECTORY_LOUVAIN, "genre_distr_2.0")

# aggiunge le community trovate con louvain alla lista dei nodi
# add_community_label(load_communities("OutputLOUVAIN/2_louvain_weighted.txt"), "nodesWithCommunity.csv")
# add_award_category_label("nodesWithAward_category.csv")

######################################### DEMON Analysis #########################################################

# # execute 101 DEMON attempts
# epsilon_start = 0
# delta = 0.01
# attempt = 1
# find_communities_demon(path_actor_network, epsilon_start, delta, attempt, MIN_COMMUNITY_SIZE_DEMON, OUTPUT_DIRECTORY_DEMON)

# # number of community plot
# number_of_communities_plot(out="number_of_community_distribution", log_directory=OUTPUT_DIRECTORY_DEMON, label_y_axis="Epsilon")

# list_communities = read_all_CD_algorithm_output_directory(log_directory=OUTPUT_DIRECTORY_DEMON)
# epsilon = 0.32
# list_communities_result = read_single_CD_Algorithm_result(RESULT_DIRECTORY, "demon_result_0.32_3.csv")
# list_communities = read_single_CD_algorithm_attempt(OUTPUT_DIRECTORY_DEMON, "demon_actor_25_0.25_3.txt")
# list_communities = read_single_CD_algorithm_attempt(OUTPUT_DIRECTORY_KCLIQUE, "4_clique_cut3.txt")
# list_communities = read_single_CD_algorithm_attempt(OUTPUT_DIRECTORY_LOUVAIN, "3_louvain_cut3.txt")
# community_analysis(list_communities[epsilon])

# for epsilon in list_communities:
#     plot_communities_length_distribution(list_communities[epsilon], "Distribution community size Epsilon: "+str(epsilon), PLOT_DIRECTORY_DEMON, "community_size_distribution_"+str(epsilon))

# list_communities = read_all_demon_directory()
# oscar_community = evaluate_CD_algorithm_attempt(list_communities, RESULT_DIRECTORY_DEMON, DEMON, MIN_COMMUNITY_SIZE_DEMON)
# # Plot overlap distribution of every attempt of different parameter in community discovery
# for epsilon in oscar_community:
#     print "Numero di Community del Successo: "+str(len(oscar_community[epsilon]))
#     plot_overlap_distribution(oscar_community[epsilon], PLOT_DIRECTORY_DEMON, epsilon)

# # Plot overlap distribution of every attempt of different parameter in community discovery
# for epsilon in list_communities:
#     plot_overlap_distribution(list_communities[epsilon], PLOT_DIRECTORY_DEMON, epsilon)

# I read the result file of the execution of community discover algorithm
# list_communities_result = read_all_demon_directory_result(log_directory=RESULT_DIRECTORY)
# now in global_statistics I have all the statistic for every parameter with I execute community discovery task (epsilon in DEMON or K in k-clique)
# global_statistics = compute_global_statistics_DEMON_attempt(list_communities_result)
# horizontal_barchar(global_statistics["labels"][epsilon], PLOT_DIRECTORY_DEMON, "genre_distribution_community_"+str(epsilon).replace(".","_"))
# # print on console the list of different label that describe the different communities found with a certain epsilon
# for key, value in sorted(global_statistics["unique_labels"].iteritems(), key=lambda (k, v): len(v)):
#     print "Epsilon: "+str(key)+" Unique Label: "+str(value)

# # plotting of the average community legnth for every k
# plot_general_barchart(global_statistics["mean_lenght"].iteritems(), PLOT_DIRECTORY_DEMON, "Epsilon", "Average Community length", "Arithmetic Density Distribution DEMON communities", "mean_length_distribution_demon_communities")
#
# # plotting of the ponderate density over every epsilon
# plot_general_barchart(global_statistics["arithmetic_density"].iteritems(), PLOT_DIRECTORY_DEMON, "Epsilon", "Mean Density", "Arithmetic Density Distribution DEMON communities", "arithmetic_density_distribution_demon_communities")
#
# # plotting of the ponderate purity over every epsilon
# plot_general_barchart(sorted(global_statistics["ponderate_purity"], PLOT_DIRECTORY_DEMON, "Epsilon", "Ponderate Mean Purity", "Purity Distribution DEMON based on Actor Genre", "purity_distribution_demon_communities_on_genre", highlight=32)

# # plot the length of the different community in a specific epsilon
# for epsilon in global_statistics["lenghts"]:
#     plot_communities_length_distribution(global_statistics["lenghts"][epsilon], "Distribution community size Epsilon: "+str(epsilon), PLOT_DIRECTORY_DEMON, "-1_community_size_distribution_"+str(epsilon))
