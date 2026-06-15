import streamlit as st
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset
movies = pd.read_csv("movies.csv")

# Create vectors
cv = CountVectorizer()
vectors = cv.fit_transform(movies['genres']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)


# Recommendation function
def recommend(movie_name):
    if movie_name not in movies['title'].values:
        return ["Movie not found!"]

    movie_index = movies[movies['title'] == movie_name].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for movie in movie_list:
        movie_title = movies.iloc[movie[0]].title
        recommendations.append(movie_title)

    return recommendations


# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.selectbox(
    "Select Movie",
    movies['title'].values
)

if st.button("Recommend"):
    results = recommend(movie_name)

    st.success("Recommendations Generated!")
    st.subheader("🎬 Recommended Movies")

    for i, movie in enumerate(results):
        st.markdown(f"### {movie}")

     