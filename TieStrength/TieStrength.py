__author__ = 'Trappola'

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys


# plot_tie_strength_vs_neighbor_overlap(weights, neighbors_overlap)
def plot_tie_strength_vs_neighbor_overlap(graph, out):
    tie_strength = []
    neighbor_overlap = []
    for o, d, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'], data['NOverlap'])): # reverse=True per ordinare al contrario
        tie_strength.append(data["weight"])
        neighbor_overlap.append(data['NOverlap'])
    avg_x, avg_y = avg_points(tie_strength, neighbor_overlap)
    plt.plot(tie_strength, neighbor_overlap, ".", markersize=3, label='<O> single point')
    plt.plot(avg_x, avg_y, "rD", markersize=5, label='<O> points same Strength')
    # plt.title("Tie Strength vs Neighbor Overlap", fontsize=15)
    plt.xlabel("Tie Strength (Number of Collaboration)", fontsize=10, labelpad=0)
    plt.ylabel("Neighbor Overlap <O>", fontsize=10, labelpad=0)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    # plt.loglog()
    plt.legend(numpoints=1, loc=0, fontsize="x-small")
    plt.savefig(out+".jpg", bbox_inches="tight")
    plt.show()


def avg_points(x, y):
    xbins = np.unique(x).tolist()                                # valori unici di degree (asse x)
    xbins.sort()
    xbins.append(max(xbins)+1)                                   # barbatrucco per avere l'ultimo bin
    n, bin_edgesX = np.histogram(x, bins=xbins)                  # n = lista con numero di punti per ogni bin ( => per ogni valore delle ascisse)
    sum_y, bin_edgesY = np.histogram(x, bins=xbins, weights=y)   # sum_y = lista con somma dei valori Y per ogni bin
    y_avg_values = sum_y / n                                     # valore Y medio per ogni bin ( => valori ordinate medi)
    xbins = np.unique(x).tolist()                                # per togliere ultimo bin aggiunto inizialmente (barbatrucco)
    return xbins, y_avg_values


def plot_link_removing_vs_len_largest_component(data):
    link_removed = []
    len_largest_component = []

    for line in data:
        item = line.rstrip().split(",")
        link_removed.append(item[0])
        len_largest_component.append(item[1])

    plt.plot(link_removed, len_largest_component, ".", markersize=5, label='<O> single point')
    # plt.title("Removed Links vs Size of largest component", fontsize=15)
    plt.xlabel("Removed Links", fontsize=10, labelpad=0)
    plt.ylabel("Size of largest component", fontsize=10, labelpad=0)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    # plt.loglog()
    plt.legend(numpoints=1, loc=0, fontsize="x-small")
    plt.show()


def plot_link_removing_vs_len_largest_component(data_weak, data_strong, out):
    link_removed_weak = []
    len_largest_component_weak = []

    for line in data_weak:
        item = line.rstrip().split(",")
        link_removed_weak.append(item[0])
        len_largest_component_weak.append(item[1])

    link_removed_strong = []
    len_largest_component_strong = []

    for line in data_strong:
        item = line.rstrip().split(",")
        link_removed_strong.append(item[0])
        len_largest_component_strong.append(item[1])

    plt.plot(link_removed_weak, len_largest_component_weak, "-", linewidth=2, label='Weak tie first')
    plt.plot(link_removed_strong, len_largest_component_strong, "r-", linewidth=2, label='Strong tie first')
    # plt.title("Removed Links vs Size of largest component", fontsize=15)
    plt.xlabel("Removed Links", fontsize=10, labelpad=0)
    plt.ylabel("Size of largest component", fontsize=10, labelpad=0)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    # plt.loglog()
    plt.legend(numpoints=1, loc=0, fontsize="x-small")
    plt.savefig(out+".jpg", bbox_inches="tight")
    plt.show()


def calculate_neighbors_overlap(graph):
    count_local_bridge = 0
    for o, d, data in graph.edges(data=True):
        neighbors_o = graph.neighbors(o)
        neighbors_d = graph.neighbors(d)
        neighbors_intersection = set(neighbors_o).intersection(neighbors_d)
        neighbors_union = set(neighbors_o).union(neighbors_d)
        node_set = set()
        node_set.add(o)
        node_set.add(d)
        neighbors_union = neighbors_union - node_set
        neighbors_overlap_coefficient = float(len(neighbors_intersection))/float(len(neighbors_union))
        if neighbors_overlap_coefficient == 0.0:
            count_local_bridge += 1
        data["NOverlap"] = neighbors_overlap_coefficient

    print "The Number of Local Bridges is: " +str(count_local_bridge)
    # write network on file
    out_file = open("actor_network_weighted_overlap.csv", "w")
    nx.write_edgelist(graph, out_file, delimiter=",", data=('weight', 'NOverlap'))

# procedure to eliminate weak link first
def weak_tie_first_removing(graph, step, cut):
    weak_tie_first_output = "weak_tie_first_output_"+str(step)+"_"+str(cut)+".csv"
    out_weak = open(weak_tie_first_output, "w")
    count_link_removed = 0
    for o, d, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'], data['NOverlap'])):
        graph.remove_edge(o, d)
        # print data["weight"]
        count_link_removed += 1
        if count_link_removed % step == 0:
            largest_component = max(nx.connected_component_subgraphs(graph), key=len)
            print str(data["weight"])+" "+str(count_link_removed)+" "+str(len(largest_component))
            res = "%s,%s\n" % (count_link_removed, len(largest_component))
            out_weak.write("%s" % res.encode('utf-8'))
            out_weak.flush()
    out_weak.close()


def strong_tie_first_removing(graph, step, cut):
    strong_tie_first_output = "strong_tie_first_output_"+str(step)+"_"+str(cut)+".csv"
    strong_weak = open(strong_tie_first_output, "w")
    count_link_removed = 0
    for o, d, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'], data['NOverlap']), reverse=True):
        graph.remove_edge(o, d)
        count_link_removed += 1
        if count_link_removed % step == 0:
            largest_component = max(nx.connected_component_subgraphs(graph), key=len)
            print str(data["weight"])+" "+str(count_link_removed)+" "+str(len(largest_component))
            res = "%s,%s\n" % (count_link_removed, len(largest_component))
            strong_weak.write("%s" % res.encode('utf-8'))
            strong_weak.flush()
    strong_weak.close()

path = "../DATA/Network_data_final/actor_network_weighted.csv"
graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float),))
calculate_neighbors_overlap(graph)

# cut = "_cut3"
# cut = ""
# path = "actor_network_weighted_overlap"+str(cut)+".csv"
# graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float), ('NOverlap', float)))
# plot_tie_strength_vs_neighbor_overlap(graph, out="Plot/tie_strength_vs_neighbor_overlap"+str(cut))

# cut = "_cut3"
# path = "actor_network_weighted_overlap"+str(cut)+".csv"
# graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float), ('NOverlap', float)))
# step = 100
# weak_tie_first_removing(graph, step, cut)

# cut = "_cut3"
# path = "actor_network_weighted_overlap"+str(cut)+".csv"
# graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float), ('NOverlap', float)))
# step = 100
# strong_tie_first_removing(graph, step, cut)

# cut = "_cut3"
# step = 50
# # path_weak = "weak_tie_first_output_"+str(step)+"_"+str(cut)+".csv"
# # path_strong = "strong_tie_first_output_"+str(step)+"_"+str(cut)+".csv"
# path_weak = "weak_tie_first_output_"+str(step)+".csv"
# path_strong = "strong_tie_first_output_"+str(step)+".csv"
# weak_tie_first = open(path_weak)
# strong_tie_first = open(path_strong)
# # plot_link_removing_vs_len_largest_component(weak_tie_first)
# # plot_link_removing_vs_len_largest_component(weak_tie_first, strong_tie_first, out="Plot/link_removing_vs_len_largest_component"+str(cut))
# plot_link_removing_vs_len_largest_component(weak_tie_first, strong_tie_first, out="Plot/link_removing_vs_len_largest_component")