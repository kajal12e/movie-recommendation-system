import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Load posters
with open("poster_urls.json", "r") as f:
    posters = json.load(f)

# Combine features
movies["tags"] = movies["genres"] + " " + movies["keywords"]

# Convert text to vectors
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    names = []
    images = []

    for i in movies_list:
        title = movies.iloc[i[0]].title
        names.append(title)
        images.append(posters.get(title, ""))

    return names, images

# UI
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get similar recommendations")

movie = st.selectbox("Choose a movie", movies["title"].values)

if st.button("Recommend"):
    names, images = recommend(movie)

    cols = st.columns(5)

    for i in range(len(names)):
        with cols[i]:
            st.text(names[i])
            if images[i]:
                st.image(images[i])