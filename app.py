import _pickle as pickle
import streamlit as st
import requests
import base64
import lzma

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    backdrop-filter: blur(15px);
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('assets/img/movie_bg_1.jpg')

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender System')
url1 = 'https://raw.githubusercontent.com/HarrisonPaulR/movie-recommender-system/blob/398310beb93ca737300d087d67f1cc3cc9b025c7/assets/pickles/movies.lzma'
url2 = 'https://raw.githubusercontent.com/HarrisonPaulR/movie-recommender-system/blob/70da3844214c4a71d881e72bca6fe8752cecf7b2/assets/pickles/similarity.lzma'
# movie_response = requests.get(url1).content
# similarity_response = requests.get(url2).content
# movies = pickle.load(movie_response,'lzma')
# similarity = pickle.load(similarity_response,'lzma')

movie_response = requests.get(url1)
similarity_response = requests.get(url2)

with lzma.open('assets/pickles/movies.lzma', 'rb') as f:
    movies = pickle.load(f)

with lzma.open('assets/pickles/similarity.lzma','rb') as f:
    similarity = pickle.load(f)

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])




