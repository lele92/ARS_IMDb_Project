__author__ = 'Matteo Borghi, Raffaele Giannella'
__license__ = "GPL"
__email__ = "matteo.borghi20@gmail.com, raph.giannella@gmail.com"

from imdb import IMDb
import os
import collections
import sys

# Instruction to crawl the data from the website
imdb_access = IMDb()
# Instruction to crawl the data on your database
#imdb_access = imdb.IMDb('sql', uri='URI_TO_YOUR_DB')

def get_collaborators(actorId, c):
    col = {}
    actor_object = imdb_access.get_person(actorId)
    role = 'actress'
    if 'actor' in actor_object.keys():
        role = 'actor'
    for movie in actor_object[role]:
        imdbMovieId = movie.movieID
        movie_object = imdb_access.get_movie(str(imdbMovieId))
        if movie_object['kind'] == "movie" or movie_object['kind'] == "video movie":
            for actor in movie_object['cast']:
                imdbPersonId = actor.personID
                variable_to_check = imdbPersonId
                if variable_to_check != actorId:
                    if str(variable_to_check) in col:
                        col[str(variable_to_check)] += 1
                    else:
                        col[str(variable_to_check)] = 1
        # Extract all Collaborator with two or more films in common
        col2 = {x: col[x] for x in col if col[x] > 1}
        c += 1
        return col2, c


# search ID of our seed Leonardo DiCaprio in our databases
# for person in ia.search_person('Leonardo DiCaprio'):
#	print person.personID, person['name']
#	person_of_leo = person

# The ID of Leonardo DiCaprio is 0000138 for Online version of IMDb
seed = 'Id of Leonardo Di Caprio'

seen = {}
seen[seed] = None
user_list = [seed]
count = 0
max_users = 5500

out = open("data%sactor_network.csv" % (os.sep, count), "w")
outw = open("data%sactor_network_weighted.csv" % (os.sep, count), "w")

while count < max_users:
    collaborators, count = get_collaborators(user_list[count], count)
    print count - 1, user_list[count - 1]
    for col in collaborators:
        res = "%s,%s\n" % (user_list[count - 1], col)
        resw = "%s,%s,%s\n" % (user_list[count - 1], col, collaborators[col])
        out.write("%s" % res.encode('utf-8'))
        outw.write("%s" % resw.encode('utf-8'))
        if col not in seen:
            seen[col] = None
            user_list.append(col)
        out.flush()
out.close()
