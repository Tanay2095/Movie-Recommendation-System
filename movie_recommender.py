import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

cv = CountVectorizer()

vectors = cv.fit_transform(
    movies['genres']
).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie_name):

    if movie_name not in movies['title'].values:
        print("\nMovie not found!")
        return

    movie_index = movies[
        movies['title'] == movie_name
    ].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:\n")

    for i, movie in enumerate(movie_list, start=1):
        print(f"{i}. {movies.iloc[movie[0]].title}")

movie_name = input("Enter movie name: ")
recommend(movie_name)



