import streamlit as st
import pandas as pd
import ast
import requests
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# ⚙️ CONFIGURATION
# =========================================================
TMDB_API_KEY = '1d3e98627e79321f7093a1b46fe360d7'
st.set_page_config(page_title="CineMatch AI", page_icon="🍿", layout="wide")

# =========================================================
# 🎨 CUSTOM STYLING
# =========================================================
st.markdown("""
<style>
    .stApp { background-color: #141414; color: #ffffff; }
    h1 { color: #E50914; font-family: 'Arial Black', sans-serif; text-align: center; }
    .movie-title { font-size: 16px; font-weight: bold; margin-bottom: 5px; }
    .rating { color: #f5c518; font-weight: bold; font-size: 14px; }
    .stButton>button { background-color: #333; color: white; border: 1px solid #555; }
    .stButton>button:hover { border-color: #E50914; color: #E50914; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🧠 SMART DATA LOADING (Cloud Compatible)
# =========================================================
@st.cache_resource
def load_and_process_data():
    # If we are on the cloud, the pkl files might not exist. 
    # We generate them on the fly.
    if not os.path.exists('similarity.pkl'):
        
        movies = pd.read_csv('data/tmdb_5000_movies.csv')
        credits = pd.read_csv('data/tmdb_5000_credits.csv')
        movies = movies.merge(credits, on='title')
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        movies.dropna(inplace=True)

        def convert(obj): return [i['name'] for i in ast.literal_eval(obj)]
        def convert3(obj): return [i['name'] for i in ast.literal_eval(obj)][:3]
        def fetch_director(obj): return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']

        movies['genres'] = movies['genres'].apply(convert)
        movies['keywords'] = movies['keywords'].apply(convert)
        movies['cast'] = movies['cast'].apply(convert3)
        movies['crew'] = movies['crew'].apply(fetch_director)
        movies['overview'] = movies['overview'].apply(lambda x: x.split())

        for col in ['genres', 'keywords', 'cast', 'crew']:
            movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        new_df = movies[['movie_id', 'title', 'tags']].copy()
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(new_df['tags']).toarray()
        similarity = cosine_similarity(vectors)
        return new_df, similarity
    else:
        # Load from local files if they exist (Faster for local dev)
        new_df = pickle.load(open('movies.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return new_df, similarity

movies, similarity = load_and_process_data()

# =========================================================
# 🌐 API FUNCTIONS (Trailers & Posters)
# =========================================================
def fetch_details(movie_id):
    try:
        # 1. Get Details (Poster + Rating)
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url).json()
        poster = "https://image.tmdb.org/t/p/w500/" + data.get('poster_path', '')
        rating = data.get('vote_average', 0)
        
        # 2. Get Trailer
        video_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
        video_data = requests.get(video_url).json()
        trailer = "None"
        
        if 'results' in video_data:
            for video in video_data['results']:
                if video['site'] == "YouTube" and video['type'] == "Trailer":
                    trailer = f"https://www.youtube.com/watch?v={video['key']}"
                    break
                    
        return poster, rating, trailer
    except:
        return "https://via.placeholder.com/500x750", 0, "None"

def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        results = []
        for i in movies_list:
            movie_id = movies.iloc[i[0]].movie_id
            title = movies.iloc[i[0]].title
            poster, rating, trailer = fetch_details(movie_id)
            results.append((title, poster, rating, trailer))
        return results
    except:
        return []

# =========================================================
# 🖥️ UI INTERFACE
# =========================================================
st.title("🍿 CineMatch AI")

# Sidebar
st.sidebar.markdown("### 🔎 Search")
selected_movie = st.sidebar.selectbox("Pick a movie you love:", movies['title'].values)
search_clicked = st.sidebar.button("Recommend Movies", type="primary")

if search_clicked:
    st.subheader(f"Because you watched '{selected_movie}':")
    recommendations = recommend(selected_movie)
    
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        title, poster, rating, trailer = recommendations[idx]
        
        with col:
            st.image(poster, use_container_width=True)
            st.markdown(f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='rating'>⭐ {round(rating, 1)}/10</div>", unsafe_allow_html=True)
            
            if trailer != "None":
                st.link_button("▶ Trailer", trailer)
            else:
                st.button("No Trailer", disabled=True, key=idx)
else:
    st.info("👈 Select a movie from the sidebar to get started!")