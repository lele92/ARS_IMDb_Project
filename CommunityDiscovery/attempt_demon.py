__author__ = 'Trappola'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com"

import Demon

print "-------------------------------------"
print "              {DEMON}                "
print "     Democratic Estimate of the      "
print "  Modular Organization of a Network  "
print "-------------------------------------\n"

network_file = "../DATA/Network_data_final/actor_network_cleaned.csv"
epsilon = 0.25
min_community_size = 1
dm = Demon.Demon(network_file, epsilon=epsilon, min_community_size=min_community_size, file_output="OutputDEMON/demon_actor_"+str(epsilon).replace(".","-")+"_"+str(min_community_size).replace(".","-"))
dm.execute()