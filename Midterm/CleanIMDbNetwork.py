__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

import os

f = open("data%sactor_network.csv" % os.sep)
out = open("data%actor_network_cleaned.csv"  % os.sep, "w")
ouu = open("data%users.csv"  % os.sep, "w")

users = {}
users_combined = {}
for l in f:
    users[l.split(",")[0]] = None
array_user = []

f = open("data%snetwork.csv" % os.sep)
for l in f:
    u = l.rstrip().split(",")
    user1 = u[0]
    user2 = u[1]
    users_combination = u[1]+u[0]
    if user1 in users and user2 in users and users_combination not in users_combined:
        out.write(l)
        users_combined[user1+user2] = None
        out.flush()
out.close()

for u in users:
    ouu.write("%s\n" % u)
ouu.flush()
ouu.close()
