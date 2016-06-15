__author__ = 'Trappola'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com"

from Demon import Demon

def find_communities_demon(network_path, epsilon_start, delta, attempt, min_community_size):
    attempt_range = range(0, attempt)
    epsilon = epsilon_start
    for i in attempt_range:
        file_output = "OutputDEMON/demon_actor_"+str(i)+"_"+str(epsilon)+"_"+str(min_community_size)
        dm = Demon(network_path, epsilon=epsilon, min_community_size=min_community_size, file_output=file_output)
        dm.execute()
        epsilon += delta

network_path = "../DATA/Network_data_final/actor_network_cleaned.csv"
epsilon_start = 0.61
delta = 0.01
attempt = 40
min_community_size = 3  # per ora delta di default dell'algoritmo
find_communities_demon(network_path, epsilon_start, delta, attempt, min_community_size)
