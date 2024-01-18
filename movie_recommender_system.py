import ast
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
# contains movie info
movies_path = "https://github.com/Harrisonpaul69/movie-recommender-system/blob/8fbf6aab5a1e4cb3b24d75c61dd37136887be2a0/data/tmdb_5000_movies.csv?raw=true"
movies = pd.read_csv(movies_path,encoding='latin1')
# contains cast and crew info
credits_path = "https://github.com/Harrisonpaul69/movie-recommender-system/blob/8fbf6aab5a1e4cb3b24d75c61dd37136887be2a0/data/tmdb_5000_credits.csv?raw=true"
credit = pd.read_csv(credits_path,encoding='latin1')
# merging movies and credits
movies = movies.merge(credit, on='title')
movies.info()

movies = movies[['title', 'genres', 'id', 'keywords', 'overview', 'cast', 'crew']]
movies.isnull().sum()
# print(movies.shape)

# dropping those rows which has null values
movies.dropna(inplace=True)

def derive(a):
    L = []
    for i in ast.literal_eval(a):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(derive)
movies['keywords'] = movies['keywords'].apply(derive)

# To get top 3 cast of the movie
def derive3(a):
    L = []
    counter = 0
    for i in ast.literal_eval(a):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(derive3)

# To get the director name alone from cast json format
def deriveDirector(a):
    L = []
    for i in ast.literal_eval(a):
        if i['job'] == "Director":
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(deriveDirector)

# Converting overview as str to list
movies['overview'] = movies['overview'].apply(lambda x: x.split(" "))

# Creating a new column by concatenating all columns
movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview'] + movies['cast'] + movies['crew']

# Storing it in a new df
new_movies = movies[['id', 'title', 'tags']].copy()

# Deleting the space between words
new_movies['tags'] = new_movies['tags'].apply(lambda x: [i.replace(" ", "") for i in x])

pd.set_option('max_colwidth', None)

# Converting tags --> list to string
new_movies['tags'] = new_movies['tags'].apply(lambda x:" ".join(x))

new_movies['tags'] = new_movies['tags'].str.lower()
# ====Preprocessing ends --Converting string to lowercase

# Vectorization starts----
# 1.Create a vector and its count using countvectorizer and remove stop words from sklearn
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new_movies['tags']).toarray()

# 2.analyse the feature name of the vectors
cv.get_feature_names_out()

# 3.remove the repeated words using stem fn from ntlk
ps = PorterStemmer()
def stemming(text):
    L =[]
    for i in text:
        L.append(ps.stem(i))
    return " ".join(L)

# 4.find the cosine similarity
similarity = cosine_similarity(vector)

# 5.write fn to calculate movie index,distance and get list of movies(top 5) with high similiarity
def recommend(movie):
    movie_index = new_movies[new_movies['title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[movie_index])),reverse=True,key=lambda x:x[1])
    for i in distance[1:6]:
        print(new_movies.iloc[i[0]].title)

