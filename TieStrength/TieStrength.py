__author__ = 'Trappola'

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_tie_strength_vs_neighbor_overlap(tie_strength, neighbor_overlap):
    avg_x, avg_y = avg_points(tie_strength, neighbor_overlap)
    plt.plot(tie_strength, neighbor_overlap, ".", markersize=3, label='CC single point')
    plt.plot(avg_x, avg_y, "rD", markersize=5, label='CC points same K')
    plt.title("Tie Strength vs Neighbor Overlap", fontsize=15)
    plt.xlabel("Tie Strength", fontsize=10, labelpad=-2)
    plt.ylabel("Neighbor Overlap", fontsize=10, labelpad=-2)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    # plt.loglog()
    plt.legend(numpoints=1, loc=0, fontsize="x-small")
    plt.show()


def avg_points(x, y):
    xbins = np.unique(x).tolist()                                # valori unici di degree (asse x)
    xbins.sort()
    xbins.append(max(xbins)+1)                                   # barbatrucco per avere l'ultimo bin
    n, bin_edgesX = np.histogram(x, bins=xbins)                  # n = lista con numero di punti per ogni bin ( => per ogni valore delle ascisse)
    sum_y, bin_edgesY = np.histogram(x, bins=xbins, weights=y)   # sum_y = lista con somma dei valori Y per ogni bin
    y_avg_values = sum_y / n                                     # valore Y medio per ogni bin ( => valori ordinate medi)
    # i = 0
    # while i < len(sum_y):
    #     print "[ "+str(bin_edgesY[i])+" - "+str(bin_edgesY[i+1])+" ] ->  NumNodi: "+str(n[i])+"\tSumClusteringCoefficient: "+str(sum_y[i])+"\tAvgClusteringCoefficient: "+str(y_avg_values[i])
    #     i += 1
    xbins = np.unique(x).tolist()                                # per togliere ultimo bin aggiunto inizialmente (barbatrucco)

    return xbins, y_avg_values


def plot_link_removing_vs_len_largest_component(link_removed, len_largest_component):
    plt.plot(link_removed, len_largest_component, ".", markersize=5, label='CC single point')
    plt.title("Removed Links vs Size of largest component", fontsize=15)
    plt.xlabel("Removed Links", fontsize=10, labelpad=-2)
    plt.ylabel("Size of largest component", fontsize=10, labelpad=-2)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    # plt.loglog()
    plt.legend(numpoints=1, loc=0, fontsize="x-small")
    plt.show()


path = "../DATA/Network_data_final/actor_network_weighted.csv"
graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float),))

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
    if int(neighbors_overlap_coefficient) == 0:
        count_local_bridge += 1
    data["NOverlap"] = neighbors_overlap_coefficient

print "The Number of Local Bridges is: " +str(count_local_bridge)

# procedure to plot strength vs Neighbor Overlap
weights = []
neighbors_overlap = []
for o, d, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'], data['NOverlap'])): # reverse=True per ordinare al contrario
    weights.append(data["weight"])
    neighbors_overlap.append(data['NOverlap'])

# plot_tie_strength_vs_neighbor_overlap(weights, neighbors_overlap)

# procedure to eliminate weak link first
weak_tie_first_output = "weak_tie_first_output.csv"
out_weak = open(weak_tie_first_output, "w")
link_removed = []
len_largest_component = []
count_link_removed = 0
for o, d, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'], data['NOverlap'])):
    graph.remove_edge(o, d)
    # print data["weight"]
    count_link_removed += 1
    link_removed.append(count_link_removed)
    largest_component = max(nx.connected_component_subgraphs(graph), key=len)
    len_largest_component.append(len(largest_component))
    print str(data["weight"])+" "+str(count_link_removed)+" "+str(len(largest_component))
    res = "%s,%s\n" % (count_link_removed, len(largest_component))
    out_weak.write("%s" % res.encode('utf-8'))
    out_weak.flush()
    # sys.exit()
out_weak.close()

strong_tie_first_output = "strong_tie_first_output.csv"
strong_weak = open(weak_tie_first_output, "w")
link_removed = []
len_largest_component = []
count_link_removed = 0
for o, d, data in sorted(graph.edges(data=True), key=lambda (a, b, data): (data['weight'], data['NOverlap']), reverse=True):
    graph.remove_edge(o, d)
    # print data["weight"]
    count_link_removed += 1
    link_removed.append(count_link_removed)
    largest_component = max(nx.connected_component_subgraphs(graph), key=len)
    len_largest_component.append(len(largest_component))
    print str(data["weight"])+" "+str(count_link_removed)+" "+str(len(largest_component))
    res = "%s,%s\n" % (count_link_removed, len(largest_component))
    strong_weak.write("%s" % res.encode('utf-8'))
    strong_weak.flush()
    # sys.exit()
strong_weak.close()

plot_link_removing_vs_len_largest_component(link_removed, len_largest_component)

