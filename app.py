import streamlit as st
import pandas as pd
import requests
from movie_recommender_system import movies,similarity

movies_list = movies
# movies_list = movies_list['title'].values
movies_list = pd.DataFrame(movies_list)


def fetch_poster(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=512018e0fe1cee7536d87fd99b001742&language=en-US".format(
            movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


def recommend(movie):
    movie_index = movies_list[movies_list['title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    posters = []
    for i in distance[1:6]:
        recommended_movies.append(movies_list.iloc[i[0]].title)
        movie_id = movies_list.iloc[i[0]].id
        posters.append(fetch_poster(movie_id))
    return recommended_movies, posters


st.title("Movie Recommender")

selected_movie = st.selectbox(
    'Enter the name of movie',
    movies_list['title'].values)

if st.button('Recommend', type="primary"):
    names, posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        image = st.image(posters[0])
        st.text(names[0])

    with col2:
        st.image(posters[1])
        st.text(names[1])

    with col3:
        st.image(posters[2])
        st.text(names[2])

    with col4:
        st.image(posters[3])
        st.text(names[3])

    with col5:
        st.image(posters[4])
        st.text(names[4])
