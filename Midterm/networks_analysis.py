# -*- coding: utf-8 -*-
__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

import networkx as nx
import matplotlib.pyplot as plt
import operator
from operator import itemgetter
import numpy as np


# method that compute the basic statistic about a graph taken as input of the function
def base_stats(graph):
    nodes = len(graph.nodes())
    edges = len(graph.edges())
    density = nx.density(graph)
    c_coefficient = nx.average_clustering(graph)
    avg_degree = 2*float(edges)/float(nodes)
    network_diameter = nx.diameter(graph)
    avg_shortest_path_length = nx.average_shortest_path_length(graph)

    print "avg_degree: " + str(avg_degree)
    print "nodes: " + str(nodes) + "\n" + \
          "edges: " + str(edges) + "\n" +\
          "density: " + str(density) + "\n" +\
           "network_diameter: " + str(network_diameter) + "\n" +\
          "c_coefficient: " + str(c_coefficient) + "\n" +\
          "avg_shortest_path_length: " + str(avg_shortest_path_length)


def largest_hub_in_graph(G):
    # find node with largest degree
    node_and_degree = G.degree()
    (largest_hub, degree) = sorted(node_and_degree.items(), key=itemgetter(1))[-1]
    # Create ego graph of main hub
    hub_ego = nx.ego_graph(G, largest_hub)
    hub_degree = nx.degree(G, largest_hub)
    print "largest hub ID: " + str(largest_hub) + "\t" + "largest hub degree: " + str(hub_degree)
    # Draw graph
    # pos = nx.spring_layout(hub_ego)
    # nx.draw(hub_ego,pos,node_color='b',node_size=50,with_labels=False)
    # Draw ego as large and red
    # nx.draw_networkx_nodes(hub_ego,pos,nodelist=[largest_hub],node_size=300,node_color='r')
    # plt.savefig('ego_graph.png')
    # plt.show()


def degree_distribution(g, title):
    # get the degree histogram
    hist = nx.degree_histogram(g)

    plt.plot(range(0, len(hist)), hist, ".", markersize=10)
    plt.title(title, fontsize=14)
    plt.xlabel("Degree", fontsize=10, labelpad=-2)
    plt.ylabel("#Nodes", fontsize=10, labelpad=-2)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.loglog()
    plt.show()


def degree_distribution_comparison(g, g1, title, model_name):
    # get the degree histogram
    hist = nx.degree_histogram(g)
    hist1 = nx.degree_histogram(g1)
    plt.plot(range(0, len(hist)), hist, ".", markersize=10, label="Actor Network")
    plt.plot(range(0, len(hist1)), hist1, "r.", markersize=10, label=model_name)
    plt.title(title, fontsize=15)
    plt.xlabel("Degree", fontsize=10, labelpad=-2)
    plt.ylabel("#Nodes", fontsize=10, labelpad=-2)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.loglog()
    plt.legend(numpoints=1, loc=0, fontsize="x-small")
    plt.show()


def get_centrality(g):
    # compute centrality
    bt = nx.betweenness_centrality(g)

    # order nodes by decreasing centrality
    bt_sorted = sorted(bt.items(), key=operator.itemgetter(1), reverse=True)    
    # Print the first 100 results
    count = 0
    for (node, betweenness) in bt_sorted:
        if count == 99:
            break
        print node, betweenness
        count += 1


def extract_ego_networks(g, actorID):

    ego_graph = nx.ego_graph(g, actorID)
    nx.draw(ego_graph, with_labels=True)
    plt.show()


def draw_connected_components(g):
    # get the connected components
    cc = nx.connected_components(g)

    i = 0
    for c in cc:
        # extract the subgraph identifying the actual component
        sub = g.subgraph(c)

        # plot only components having at least 3 nodes
        if len(sub) > 3:
            nx.draw(sub)
            plt.show()
            print (len(sub))
            print nx.average_shortest_path_length(sub)


def actor_network_analysis(path):
    print "*********** ACTOR NETWORK ***********"
    input_actor_network = open(path)
    actor_network = nx.read_edgelist(input_actor_network, delimiter=',', nodetype=str)
    print "########## BASE STATS ##########"
    base_stats(actor_network)
    print "########## LARGEST HUB ##########"
    largest_hub_in_graph(actor_network)
    print "########## DEGREE DISTRIBUTIONS ##########"
    degree_distribution(actor_network, "Degree Distribution Actor Network")
    print "########## BETWEENNESS CENTRALITY ##########"
    get_centrality(actor_network)
    actorID_ego_network = "0000138" #DI CAPRIO
    print "########## EGONETWORK FOR " + actorID_ego_network + "##########"
    extract_ego_networks(actor_network, actorID_ego_network)
    draw_connected_components(actor_network)
    print "########## DEGREE VS CLUSTERING COEFFICIENT ##########"
    plot_degree_vs_clustering(actor_network, "Degree Vs Clustering Coefficient Actor Network")
    return actor_network


def random_network_analysis(path_random_network, actor_graph):
    # out_file = open(input_folder+"/random_network_"+str(num_node_network)+".csv", "w")
    input_random_network = open(path_random_network)
    random_network = nx.read_edgelist(input_random_network, delimiter=',', nodetype=str)
    print "*********** RANDOM NETWORK ***********"
    # g = nx.erdos_renyi_graph(num_nodes, density)
    # nx.write_edgelist(g, out_file, delimiter=",")
    print "########## BASE STATS RANDOM NETWORK ##########"
    base_stats(random_network)
    print "########## LARGEST HUB ##########"
    largest_hub_in_graph(random_network)
    print "########## DEGREE DISTRIBUTIONS RANDOM NETWORK ##########"
    degree_distribution_comparison(actor_graph, random_network, "Degree Distribution Actor Network Vs Random Network", "Random Network")
    print "########## DEGREE VS CLUSTERING COEFFICIENT ##########"
    plot_degree_vs_clustering(random_network, "Degree Vs Clustering Coefficient Random Network")


def barabasi_albert_analysis(path_barabasi_albert_network, actor_graph):
    #g = nx.barabasi_albert_graph(num_nodes, avg_degree)
    input_barabasi_albert_network = open(path_barabasi_albert_network)
    barabasi_albert_network = nx.read_edgelist(input_barabasi_albert_network, delimiter=',', nodetype=str)

    print "########## BASE STATS BARABASI ALBERT ##########"
    base_stats(barabasi_albert_network)
    print "########## LARGEST HUB ##########"
    largest_hub_in_graph(barabasi_albert_network)
    print "########## DEGREE DISTRIBUTIONS PREFERENTIAL ATTACHMENT ##########"
    degree_distribution_comparison(actor_graph, barabasi_albert_network, "Degree Distribution Actor Network Vs Barabasi Albert Network", "Barabasi Albert Network")
    print "########## DEGREE VS CLUSTERING COEFFICIENT ##########"
    plot_degree_vs_clustering(barabasi_albert_network, "Degree Vs Clustering Coefficient Barabasi Albert")


def plot_degree_vs_clustering(g, title):
    avg_clustering_coefficient = nx.average_clustering(g)
    nodes = g.nodes()
    x = []
    y = []
    for node in nodes:
        degree = nx.degree(g, node)
        clustering_coefficient = nx.clustering(g, node)
        x.append(degree)
        y.append(clustering_coefficient)

    avg_x, avg_y = avg_points(x, y)
    plt.axhline(y=avg_clustering_coefficient, linewidth=3, color="g", label="<C>")
    plt.plot(x, y, ".", markersize=3, label='CC single point')
    plt.plot(avg_x, avg_y, "rD", markersize=3, label='CC points same K')
    plt.title(title, fontsize=15)
    plt.xlabel("Degree", fontsize=10, labelpad=-2)
    plt.ylabel("Clustering Coefficient", fontsize=10, labelpad=-2)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.loglog()
    plt.legend(numpoints=1, loc=0, fontsize="x-small")
    plt.show()


def avg_points(x, y):
    xbins = np.unique(x).tolist()                                
    xbins.sort()
    xbins.append(max(xbins)+1)                                   
    n, bin_edgesX = np.histogram(x, bins=xbins)                  
    sum_y, bin_edgesY = np.histogram(x, bins=xbins, weights=y)   
    y_avg_values = sum_y / n                                     
    xbins = np.unique(x).tolist()
	
    return xbins, y_avg_values


input_folder_actor_network = "data_final/actor_network.csv"
input_folder_barabasi_albert_network = "data_final/barabasi_albert_network.csv"
input_folder_random_network = "data_final/random_network.csv"

actor_network_graph = None

actor_network_graph = actor_network_analysis(input_folder_actor_network)

random_network_analysis(input_folder_random_network, actor_network_graph)

barabasi_albert_analysis(input_folder_barabasi_albert_network, actor_network_graph)





