__author__ = 'Trappola'
# con questo file leggo i riusltati di demon e li inserisco in un dizionario cosi formattato:
# "communityLabel" : [ array contenente chiavi oggetti presenti nella community]
# il tutto è dentro l'oggetto community con cui

epsilon = 0.25
min_community_size = 50
f = open("OutputDEMON/demon_actor_"+str(epsilon).replace(".","-")+"_"+str(min_community_size).replace(".","-")+".txt")

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

# print community["63"]