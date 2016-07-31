__author__ = 'Trappola'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com"



def find_communities_demon(network_path, epsilon_start, delta, attempt, min_community_size):
    attempt_range = range(0, attempt)
    epsilon = epsilon_start
    for i in attempt_range:
        file_output = "OutputDEMON_Award/demon_actor_"+str(i)+"_"+str(epsilon)+"_"+str(min_community_size)
        dm = Demon(network_path, epsilon=epsilon, min_community_size=min_community_size, file_output=file_output)
        dm.execute()
        epsilon += delta

# network_path = "../DATA/Network_data_final/actor_network_cleaned.csv"
# network_path = "../DATA/Network_data_final/actor_network_cut3_awarded.csv"
epsilon_start = 0
delta = 0.01
attempt = 101
min_community_size = 3  # per ora delta di default dell'algoritmo
find_communities_demon(network_path, epsilon_start, delta, attempt, min_community_size)
