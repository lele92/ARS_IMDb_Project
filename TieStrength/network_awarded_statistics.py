# -*- coding: utf-8 -*-
__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

import networkx as nx
import numpy as np
import json
from collections import Counter
import sys

second_level_award = ["Golden Globe", "BAFTA Film Award", "Golden Lion", "Palme d'Or", "Golden Berlin Bear",
                     "Filmfare Award", "European Film Award", "Golden Leopard", "Grand Jury Prize", "Grand Jury Award",
                     "Grand Jury"]

oscar_valid_description = ["Best Performance by an Actor in a Leading Role",
                           "Best Performance by an Actor in a Supporting Role",
                           "Best Actor in a Leading Role",
                           "Best Actor in a Supporting Role",
                           "Best Actress in a Leading Role",
                           "Best Actress in a Supporting Role",
                           "Best Performance by an Actress in a Supporting Role",
                           "Best Performance by an Actress in a Leading Role",
                           ]


def found_year_first_oscar(awards):
    years_oscar_winning = []
    for award in awards:
        if award["type"] == "Oscar" and award["outcome"] == "Won" and award["description"] in oscar_valid_description:
            years_oscar_winning.append(int(award["year"]))
    # print min(years_oscar_winning)
    year_first_oscar = min(years_oscar_winning)
    # if year_first_oscar > 1900:
    #     # print year_first_oscar
    #     for award in awards:
    #         if award["type"] == "Oscar" and award["outcome"] == "Won" and int(award["year"]) == year_first_oscar:
    #             print award["description"]
    return min(years_oscar_winning)

def found_year_first_nomination_oscar(awards):
    years_oscar_winning = []
    for award in awards:
        if award["type"] == "Oscar" and award["description"] in oscar_valid_description:
            years_oscar_winning.append(int(award["year"]))
    # print min(years_oscar_winning)
    year_first_oscar = min(years_oscar_winning)
    # if year_first_oscar > 1900:
    #     # print year_first_oscar
    #     for award in awards:
    #         if award["type"] == "Oscar" and award["outcome"] == "Won" and int(award["year"]) == year_first_oscar:
    #             print award["description"]
    return min(years_oscar_winning)


def number_oscar_nomination(awards, year_first_oscar):
    oscar_nomination = 0
    for award in awards:
        if award["type"] == "Oscar" and award["outcome"] == "Nominated" and int(award["year"]) <= year_first_oscar:
            oscar_nomination += 1

    return oscar_nomination


def number_won_second_level_prize(awards, year_first_oscar):
    second_level_won = 0
    for award in awards:
        if award["type"] in second_level_award and award["outcome"] == "Won" and int(award["year"]) <= year_first_oscar:
            second_level_won += 1
    return second_level_won


def number_general_nomination(awards, year_first_oscar):
    number_general_nomination = 0
    for award in awards:
        if award["type"] != "Oscar" and int(award["year"]) <= year_first_oscar:
            number_general_nomination += 1
    return number_general_nomination


def oscar_winner_check(awards):
    is_oscar_winner = False
    for award in awards:
        if award["type"] == "Oscar" and award["outcome"] == "Won" and int(award["year"]) <= 2016:
            is_oscar_winner = True
    return is_oscar_winner


def assign_award_category_actor():

    actors_oscar = []
    for tmp in graph.nodes():
        for award in actors_data[tmp]["award"]:
            if award["type"] == "Oscar" and award["outcome"] == "Won":
                actors_oscar.append(tmp)
                graph.remove_node(tmp)
                actors_data[tmp]["award_category"] = "Oscar"
                actors_data[tmp]["oscar_year"] = found_year_first_oscar(actors_data[tmp]["award"])
                break

    actors_oscar_nomination = []
    for tmp in graph.nodes():
        for award in actors_data[tmp]["award"]:
            if award["type"] == "Oscar":
                actors_oscar_nomination.append(tmp)
                graph.remove_node(tmp)
                actors_data[tmp]["award_category"] = "NominationOscar"
                break

    actors_winning = []
    for tmp in graph.nodes():
        for award in actors_data[tmp]["award"]:
            if (award["type"] == "Golden Globe" and award["outcome"] == "Won") or \
                (award["type"] == "BAFTA Film Award" and award["outcome"] == "Won") or\
                (award["type"] == "Golden Lion" and award["outcome"] == "Won") or\
                ("grand jury" in award["type"] and award["outcome"] == "Won") or\
                (award["type"] == "Palme d'Or" and award["outcome"] == "Won") or\
                (award["type"] == "Golden Berlin Bear" and award["outcome"] == "Won") or\
                (award["type"] == "Filmfare Award" and award["outcome"] == "Won") or\
                (award["type"] == "European Film Award" and award["outcome"] == "Won") or\
                (award["type"] == "Golden Leopard" and award["outcome"] == "Won"):# or\
                # (award["type"] == "Primetime Emmy" and award["outcome"] == "Won"):
                actors_winning.append(tmp)
                graph.remove_node(tmp)
                actors_data[tmp]["award_category"] = "Winning"
                break

    for tmp in graph.nodes():
        actors_data[tmp]["award_category"] = "GenericNomination"

    out = open("../DATA/File_IMDb/actor_full_genre_cleaned.json", "w")
    out.write(json.dumps(actors_data, indent=4))
    out.close()

    print "Number of Actor Oscar: "+str(len(actors_oscar))
    print "Number of Actor Nominated for Oscar: "+str(len(actors_oscar_nomination))
    print "Number of Actor Winning a prestigious Award: "+str(len(actors_winning))
    print "Number of Actor with general Nomination: "+str(len(graph.nodes()))


def search_and_write_success_candidate_actor(Oscar_winner_profile):

    oscar_candidate = open("oscar_candidate.txt", 'w')
    oscar_candidate_id = open("oscar_candidate_id.txt", 'w')
    oscar_candidate_ids = []

    for tmp in graph.nodes():
        if not oscar_winner_check(actors_data[tmp]["award"]) and actors_data[tmp]["birth date"]:
            year_check = 2016
            age = year_check - int(actors_data[tmp]["birth date"])
            num_oscar_nomination = number_oscar_nomination(actors_data[tmp]["award"], year_check)
            second_level_won = number_won_second_level_prize(actors_data[tmp]["award"], year_check)
            num_generical_award = number_general_nomination(actors_data[tmp]["award"], year_check) - second_level_won

            # if age > (Oscar_winner_profile["Age"]["mean"] - Oscar_winner_profile["Age"]["std"]) and\
            #         age < (Oscar_winner_profile["Age"]["mean"] + Oscar_winner_profile["Age"]["std"]) and\
            #         num_oscar_nomination >= Oscar_winner_profile["OscarNomination"]["mean"] and\
            #         second_level_won >= Oscar_winner_profile["SecondPrizeWin"]["mean"]:
                    # num_generical_award >= Oscar_winner_profile["GenericAwardNumber"]["mean"]:

            #print Oscar_winner_profile["SecondPrizeWin"]["mean"]
            #print int(Oscar_winner_profile["SecondPrizeWin"]["mean"])
            # sys.exit()

            if age > (Oscar_winner_profile["Age"]["mean"] - Oscar_winner_profile["Age"]["std"]) and\
                    age < (Oscar_winner_profile["Age"]["mean"] + Oscar_winner_profile["Age"]["std"]) and\
                    second_level_won >= int(Oscar_winner_profile["SecondPrizeWin"]["mean"]) and\
                    num_generical_award >= Oscar_winner_profile["GenericAwardNumber"]["median"]:

                string_data = tmp +","+str(age)+","+str(num_oscar_nomination)+","+str(second_level_won)+","+str(num_generical_award)+","+actors_data[tmp]["name"]+","+actors_data[tmp]["top_director"]+","+str(actors_data[tmp]["top_directors_dict"])+"\n"
                print tmp +","+str(age)+","+str(num_oscar_nomination)+","+str(second_level_won)+","+str(num_generical_award)+","+actors_data[tmp]["name"]
                oscar_candidate.write(string_data)
                oscar_candidate_ids.append(tmp)

    oscar_candidate.close()
    oscar_candidate_id.write(str(oscar_candidate_ids))
    oscar_candidate_id.close()
    print len(oscar_candidate_ids)


def found_success_profile():
    # cut = "cut3"
    # path = "actor_network_weighted_"+str(cut)+"_awarded.csv"
    path = "../DATA/Network_data_final/actor_network_weighted.csv"
    graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float),))

    first_oscar_age = []
    first_nomination_oscar_age = []
    oscar_nominations = []
    second_level_prizes_won = []
    generical_awards = []
    registi = []

    count_salvezza = 0

    for tmp in graph.nodes():
        for award in actors_data[tmp]["award"]:
            # Enter inside the if only if found an Oscar Winner, with also the field birth date
            if award["type"] == "Oscar" and award["outcome"] == "Won" and actors_data[tmp]["birth date"] and award["description"] in oscar_valid_description:


                year_first_oscar = found_year_first_oscar(actors_data[tmp]["award"])

                if year_first_oscar <= 2010:
                    oscar_id.append(tmp)
                    age_first_oscar = year_first_oscar - int(actors_data[tmp]["birth date"])
                    first_oscar_age.append(age_first_oscar)
                    age_first_nomination_oscar = found_year_first_nomination_oscar(actors_data[tmp]["award"]) - int(actors_data[tmp]["birth date"])
                    first_nomination_oscar_age.append(age_first_nomination_oscar)
                    num_oscar_nomination = number_oscar_nomination(actors_data[tmp]["award"], year_first_oscar)
                    oscar_nominations.append(num_oscar_nomination)
                    second_level_won = number_won_second_level_prize(actors_data[tmp]["award"], year_first_oscar)
                    second_level_prizes_won.append(second_level_won)
                    num_generical_award = number_general_nomination(actors_data[tmp]["award"], year_first_oscar) - second_level_won
                    generical_awards.append(num_generical_award)
                    # print tmp+"\t"+actors_data[tmp]["name"]+"\t"+str(age_first_oscar)+"\t"+str(num_oscar_nomination)+"\t"+str(second_level_won)+"\t"+str(num_generical_award)
                    registi.append(actors_data[tmp]["top_director"])
                else:
                    age_first_oscar = year_first_oscar - int(actors_data[tmp]["birth date"])
                    age_first_nomination_oscar = found_year_first_nomination_oscar(actors_data[tmp]["award"]) - int(actors_data[tmp]["birth date"])
                    num_oscar_nomination = number_oscar_nomination(actors_data[tmp]["award"], year_first_oscar)
                    second_level_won = number_won_second_level_prize(actors_data[tmp]["award"], year_first_oscar)
                    num_generical_award = number_general_nomination(actors_data[tmp]["award"], year_first_oscar) - second_level_won
                    print tmp+"\t"+str(age_first_nomination_oscar)+"\t"+str(year_first_oscar)+"\t"+actors_data[tmp]["name"]+"\t"+str(age_first_oscar)+"\t"+str(num_oscar_nomination)+"\t"+str(second_level_won)+"\t"+str(num_generical_award)
                    count_salvezza += 1
                # if year_first_oscar > 2010:
                #     print tmp+"\t"+actors_data[tmp]["name"]+"\t"+str(age_first_oscar)+"\t"+str(num_oscar_nomination)+"\t"+str(second_level_won)+"\t"+str(num_generical_award)
                break

    print "count salvezza: "+str(count_salvezza)

    print len(first_oscar_age)
    # sys.exit()
    print min(first_oscar_age)
    print max(first_oscar_age)
    # print np.median(np.array(oscar_nominations))
    print "Mediana Generic Award: "+ str(np.median(np.array(generical_awards)))
    print "Mediana Premi di secondo livello: "+ str(np.median(np.array(second_level_prizes_won)))
    print "Mediana Oscar Nomination: "+ str(np.median(np.array(oscar_nominations)))
    first_oscar_age = np.array(first_oscar_age)
    oscar_nominations = np.array(oscar_nominations)
    second_level_prizes_won = np.array(second_level_prizes_won)
    generical_awards = np.array(generical_awards)
    registi_dict = Counter(registi)
    registi_dict = dict(registi_dict.most_common(20))

    print len(first_oscar_age)
    print "Anno Medio primo Oscar: "+str(np.mean(first_oscar_age)) +" Varianza: "+str(np.std(first_oscar_age, ddof=1))
    print "Anno Medio prima nomination Oscar: "+str(np.mean(first_nomination_oscar_age)) +" Varianza: "+str(np.std(first_nomination_oscar_age, ddof=1))
    print "Numero nomination Oscar Medio: "+str(np.mean(oscar_nominations)) +" Varianza: "+str(np.std(oscar_nominations, ddof=1))
    print "Numero Premi di secondo livello vinti medio: "+str(np.mean(second_level_prizes_won)) +" Varianza: "+str(np.std(second_level_prizes_won, ddof=1))
    print "Numero di Premi Generici: "+str(np.mean(generical_awards)) +" Varianza: "+str(np.std(generical_awards, ddof=1))

    Oscar_winner_profile = {
        "Age": {
            "mean": np.mean(first_oscar_age),
            "std": np.std(first_oscar_age, ddof=1)
        },
        "OscarNomination": {
            "mean": np.mean(oscar_nominations),
            "std": np.std(oscar_nominations, ddof=1)
        },
        "SecondPrizeWin": {
            "mean": np.mean(second_level_prizes_won),
            "std": np.std(second_level_prizes_won, ddof=1)
        },
        "GenericAwardNumber": {
            "mean": np.mean(generical_awards),
            "median": np.median(generical_awards),
            "std": np.std(generical_awards, ddof=1)
        },
        "Registi": registi_dict
    }

    return Oscar_winner_profile



path_actor = "../DATA/File_IMDb/actor_full_genre_cleaned.json"
file_actor = open(path_actor).read()
actors_data = json.loads(file_actor)

path = "../DATA/Network_data_final/actor_network_weighted.csv"
graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float),))

oscar_id = []
oscar_winner_profile = found_success_profile()

#sys.exit()

Oscar_profile_file = open("oscar_profile_v107.txt", 'w')
Oscar_profile_file.write(json.dumps(oscar_winner_profile, indent=4))
Oscar_profile_file.close()


print len(oscar_id)

count_positive = 0
for tmp in oscar_id:

    year_first_oscar = found_year_first_oscar(actors_data[tmp]["award"])
    age_first_oscar = year_first_oscar - int(actors_data[tmp]["birth date"])
    num_oscar_nomination = number_oscar_nomination(actors_data[tmp]["award"], year_first_oscar)
    second_level_won = number_won_second_level_prize(actors_data[tmp]["award"], year_first_oscar)
    num_generical_award = number_general_nomination(actors_data[tmp]["award"], year_first_oscar) - second_level_won

    # if age_first_oscar > (oscar_winner_profile["Age"]["mean"] - oscar_winner_profile["Age"]["std"]) and\
    #         age_first_oscar < (oscar_winner_profile["Age"]["mean"] + oscar_winner_profile["Age"]["std"]) and\
    #         num_oscar_nomination >= oscar_winner_profile["OscarNomination"]["mean"] and\
    #         second_level_won >= oscar_winner_profile["SecondPrizeWin"]["mean"]:

    if age_first_oscar > (oscar_winner_profile["Age"]["mean"] - oscar_winner_profile["Age"]["std"]) and\
            age_first_oscar < (oscar_winner_profile["Age"]["mean"] + oscar_winner_profile["Age"]["std"]) and\
            second_level_won >= int(oscar_winner_profile["SecondPrizeWin"]["mean"]) and\
            num_generical_award >= oscar_winner_profile["GenericAwardNumber"]["median"]:

        count_positive += 1
        string_data = tmp +","+str(age_first_oscar)+","+str(num_oscar_nomination)+","+str(second_level_won)+","+str(num_generical_award)+","+actors_data[tmp]["name"]+","+actors_data[tmp]["top_director"]+","+str(actors_data[tmp]["top_directors_dict"])+"\n"
        # print tmp +","+str(age_first_oscar)+","+str(num_oscar_nomination)+","+str(second_level_won)+","+str(num_generical_award)+","+actors_data[tmp]["name"]

print "Positive Training Set: " + str(count_positive)

search_and_write_success_candidate_actor(oscar_winner_profile)

#
# cut = "cut3"
# path = "actor_network_weighted_"+str(cut)+"_awarded.csv"
# graph = nx.read_edgelist(path, delimiter=',', nodetype=str, data=(('weight', float),))
#
# assign_award_category_actor()
