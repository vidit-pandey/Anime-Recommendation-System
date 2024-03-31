import pickle
import streamlit as st
import numpy as np

st.header('Anime Recommendation System')
model = pickle.load(open('artifacts/model.pkl', 'rb'))
anime_names = pickle.load(open('artifacts/anime_name.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
anime_pivot = pickle.load(open('artifacts/anime_pivot.pkl', 'rb'))


def fetch_score(suggestion):
    anime_name = []
    ids_index = []
    scores = []

    for anime_id in suggestion:
        anime_name.append(anime_pivot.index[anime_id])

    for name in anime_name[0]:
        ids = np.where(final_rating['Name'] == name)[0]
        if len(ids) > 0:
            ids_index.append(ids[0])

    for idx in ids_index:
        score = final_rating.iloc[idx]['Score']
        scores.append(score)

    return scores


def recommend_anime(anime_name):
    anime_list = []
    anime_id = np.where(anime_pivot.index == anime_name)[0][0]
    distance, suggestion = model.kneighbors(anime_pivot.iloc[anime_id, :].values.reshape(1, -1), n_neighbors=11)

    scores = fetch_score(suggestion)

    for i in range(len(suggestion)):
        anime = anime_pivot.index[suggestion[i]]
        for j in anime:
            anime_list.append(j)
    return anime_list, scores


selected_anime = st.selectbox(
    "Please Type or Select an anime from the dropdown",
    anime_names
)

st.markdown('[Home](http://127.0.0.1:8000/home/) | [About Us](http://127.0.0.1:8000/aboutus/)')

if st.button('Show Recommendations'):
    recommended_anime, scores = recommend_anime(selected_anime)
    sorted_recommendations = sorted(zip(recommended_anime[1:], scores[1:]), key=lambda x: x[1], reverse=True)
    for anime, score in sorted_recommendations[:10]:
        st.write(f"{anime} - Score: {score}")
