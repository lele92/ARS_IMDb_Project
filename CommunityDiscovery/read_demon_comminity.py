__author__ = 'Trappola'
# con questo file leggo i riusltati di demon e li inserisco in un dizionario cosi formattato:
# "communityLabel" : [ array contenente chiavi oggetti presenti nella community]
# il tutto è dentro l'oggetto community con cui

import os
import matplotlib.pyplot as plt
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
        print len(community[u[0]])


def read_all_demon_directory(directory_demon_path, epsilons):
    list_communities = []
    for p in epsilons:
        dict_list = os.listdir(directory_demon_path)
        for d in dict_list:
            if p in d:
                list_dict_values = list(read_demon_community(directory_demon_path+d).values())
                list_dict_values.sort(key=len, reverse=True)
                list_communities.append(list_dict_values)
    return list_communities


def histogram(x, freq, xlabel=None, ylabel=None, out=None):
    for i in range(0,len(x)-1):
        if (i%5 != 0):
            x[i] = ""

    plt.bar(range(len(freq)), freq, color='g',alpha=0.6,linewidth=0)
    plt.xticks(range(len(x)), x, size='small',rotation='vertical')

    if (xlabel != None and ylabel != None):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if out == None:
        plt.show()
    else:
        plt.savefig(out+".svg",bbox_inches="tight")


def plot_epsilon_dict(log_directory="../demon_log/", out=None):
    l = {}
    dict_list = os.listdir(log_directory)
    dict_list.sort()
    for d in dict_list:
        l[float(d.split("_")[2])] = len(dict_from_file(log_directory +d))
    x = []
    freq = []
    for i in sorted(l):
        x.append(i)
        freq.append(l[i])
    histogram(x,freq,"Epsilon","Number of communities",out)
epsilon = 0.25
min_community_size = 50
f = open("OutputDEMON/demon_actor_"+str(epsilon).replace(".","-")+"_"+str(min_community_size).replace(".","-")+".txt")


# print community["63"]