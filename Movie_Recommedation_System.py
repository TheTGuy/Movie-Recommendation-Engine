import pandas as pd
import numpy as np

dataset = pd.read_csv("movie_dataset.csv")

# we will only use the four featues for finding similarities between movies
features = ['keywords','cast','genres','director']
# replace nan values by empty string
for feature in features:
    dataset[feature] = dataset[feature].fillna('')


# function that returns the four fearures in form of a single string 
def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
data_combined = dataset.apply(combine_features,axis=1)


# converts the matrix of strings into matrix of vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(data_combined)


# finding correlation between all possible pairs of movies
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix)


def get_index_from_title(title):
    return dataset[dataset.title == title]["index"].values[0]
def get_title_from_index(index):
    return dataset[dataset.index == index]["title"].values


movie_user_likes = "The Avengers"
movie_index = get_index_from_title(movie_user_likes)
similar_movies =  list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]


print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for i in range(0,10):
    print(get_title_from_index(sorted_similar_movies[i][0]))