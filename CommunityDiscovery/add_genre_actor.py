__author__ = 'Trappola'

import json
import numpy as np
import sys
import matplotlib.pyplot as plt


movie_kind_not_admitted = ["episode", "tv series"]

# Plot distribuzione distanze in ordine crescente
# values - lista delle distanze
def plot_distances_distribution(values, title, out):
    l = [float(x) for x in values]
    l.sort()
    # print l[:20]
    plt.plot(l, "-", linewidth=2, color="m")
    # plt.title(title)
    plt.ylabel("Percentage fitting")
    plt.xlabel("Nodes (actor)")
    # plt.axis([0, 6000, 0, 110])
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.xlim([-0.5, len(values)])
    plt.ylim([0, 105])
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.savefig(out, bbox_inches="tight")
    plt.show()


def plot_genres_distribution(g_data, title, out):
    g_data = sorted(g_data.iteritems(), key=lambda (k, v): v, reverse=True)
    x_axis = []
    y_axis = []
    genres_label = []
    count = 1
    for key, value in g_data:
        x_axis.append(count)
        y_axis.append(value)
        genres_label.append(key)
        count += 1

    plt.bar(x_axis, y_axis, align='center', color="m", alpha=0.8)
    plt.xticks(x_axis, genres_label, rotation='vertical')
    # plt.title(title)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.xlim([-0.02, len(x_axis)+0.5])
    # plt.ylim([-10, 105])
    plt.gca().yaxis.grid(True)
    plt.savefig(out, bbox_inches="tight")
    plt.show()

path_actor = "../DATA/File_IMDb/actor_full_cleaned.json"
path_actor_film = "../DATA/Indexes/actors_film.csv"
path_film = "../DATA/File_IMDb/film_cleaned.json"

file_actor = open(path_actor).read()
actors_data = json.loads(file_actor)
for tmp in actors_data:
    actors_data[tmp]["film"] = []


file_film = open(path_film).read()
films_data = json.loads(file_film)

print len(films_data)
# sys.exit()

file_actor_film = open(path_actor_film)
for l in file_actor_film:
    tmp = l.rstrip().split(",")
    actors_data[tmp[0]]["film"].append(tmp[1])

# print films_data["0236172"]
# sys.exit()
top_genre_distribution = {}
top_language_distribution = {}
top_director_distribution = {}
list_film = []
list_percentage = []
count = 0
for tmp in actors_data:
    # if actors_data[tmp]["birth date"] is not None and actors_data[tmp]["award"]:
    # tmp = "0000093"
    actor_genres = {}
    actor_languages = {}
    actor_director = {}
    for film in actors_data[tmp]["film"]:
        if film in films_data:
            genres = films_data[film]["genres"]
            if genres is not None:
                for g in genres:
                    if g in actor_genres:
                        actor_genres[g] += 1
                    else:
                        actor_genres[g] = 1
            languages = films_data[film]["languages"]
            languages = np.unique(languages).tolist()
            if languages is not None:
                for l in languages:
                    if l in actor_languages:
                        actor_languages[l] += 1
                    else:
                        actor_languages[l] = 1
            # print films_data[film]["kind"]
            if films_data[film]["director"] is not None and films_data[film]["kind"] not in movie_kind_not_admitted:
                for director in films_data[film]["director"]:
                    # print l["person_id"]
                    if director["person_id"] in actor_director:
                        actor_director[director["person_id"]] += 1
                    else:
                        actor_director[director["person_id"]] = 1
        else:
            list_film.append(film)
            print film
    # print actor_director
    # sys.exit()

    # print actor_director

    actor_genres = sorted(actor_genres.iteritems(), key=lambda (k, v): v, reverse=True)[:3]
    actor_languages = sorted(actor_languages.iteritems(), key=lambda (k, v): v, reverse=True)[:3]
    actor_director = sorted(actor_director.iteritems(), key=lambda (k, v): v, reverse=True)[:5]
    actors_data[tmp]["top_genre"] = actor_genres[0][0]
    actors_data[tmp]["top_language"] = actor_languages[0][0]
    actors_data[tmp]["top_director"] = None
    actors_data[tmp]["top_directors_dict"] = dict(actor_director)
    if actor_director:
        actors_data[tmp]["top_director"] = actor_director[0][0]
    # print actors_data[tmp]["top_director"]
    # sys.exit()
    # print actors_data[tmp]["top_director"]
    # sys.exit()
    if actors_data[tmp]["top_genre"] in top_genre_distribution:
        top_genre_distribution[actors_data[tmp]["top_genre"]] += 1
    else:
        top_genre_distribution[actors_data[tmp]["top_genre"]] = 1

    if actors_data[tmp]["top_language"] in top_language_distribution:
        top_language_distribution[actors_data[tmp]["top_language"]] += 1
    else:
        top_language_distribution[actors_data[tmp]["top_language"]] = 1

    if actors_data[tmp]["top_director"] in top_director_distribution:
        top_director_distribution[actors_data[tmp]["top_director"]] += 1
    else:
        top_director_distribution[actors_data[tmp]["top_director"]] = 1

    actors_data[tmp]["top3_genre"] = []
    for key, value in actor_genres:
        actors_data[tmp]["top3_genre"].append(key)
    list_percentage.append((float(actor_genres[0][1])/float(len(actors_data[tmp]["film"])))*100)
    # list_percentage.append((float(actor_languages[0][1])/float(len(actors_data[tmp]["film"])))*100)
    # sys.exit()

    # print actor_languages
    # print "Top Languages "+str(actors_data[tmp]["top_language"])
    # print (float(actor_languages[0][1])/float(len(actors_data[tmp]["film"])))*100
    # if count == 5:
    #     sys.exit()
    # count += 1

out = open("../DATA/File_IMDb/actor_full_genre_cleaned.json", "w")
out.write(json.dumps(actors_data, indent=4))
out.close()

path_actor_final = "../DATA/File_IMDb/actor_full_genre_cleaned.json"
file_actor = open(path_actor_final).read()
actors_data = json.loads(file_actor)
for tmp in actors_data:
    if actors_data[tmp]["birth date"] is not None and len(actors_data[tmp]["birth date"]) > 4:
        tmp_data = actors_data[tmp]["birth date"].split("-")[2]
        actors_data[tmp]["birth date"] = "19" + tmp_data
        print actors_data[tmp]["birth date"]

out = open("../DATA/File_IMDb/actor_full_genre_cleaned.json", "w")
out.write(json.dumps(actors_data, indent=4))
out.close()


# print max(list_percentage)
# print min(list_percentage)
# print reduce(lambda x, y: x + y, list_percentage) / len(list_percentage)
# print len(list_percentage)
# print sorted(list_percentage)[-5:]
# print np.median(np.array(list_percentage))
# plot_genres_distribution(top_language_distribution, "Top Language Distribution", "Plot/top_language_distribution.jpg")
# plot_distances_distribution(list_percentage, "Top Language Percentage for Actor", "Plot/top_language_fitting_percentage.jpg")

# plot_genres_distribution(top_director_distribution, "Top Director Distribution", "PlotAward/top_director_distribution.jpg")
# plot_distances_distribution(list_percentage, "Top Genre Percentage for Actor", "Plot/top_genre_fitting_percentage.jpg")

# out = open("../DATA/Indexes/film_mancanti.csv", "w")
# print len(list_film)
# for film in list_film:
#     res = "%s\n" % (film)
#     out.write("%s" % res.encode('utf-8'))
# out.close()