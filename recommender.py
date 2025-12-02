import streamlit as st
import pandas as pd
import ast
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# ⚙️ 1. CONFIGURATION
# =========================================================
TMDB_API_KEY = '1d3e98627e79321f7093a1b46fe360d7'

st.set_page_config(
    page_title="CineMatch Pro", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 2. PROFESSIONAL UI STYLING (CSS)
# =========================================================
st.markdown("""
<style>
    /* MAIN BACKGROUND */
    .stApp {
        background-color: #000000;
        background-image: linear-gradient(to bottom, rgba(0,0,0,0.8), #141414), 
                          url('https://assets.nflxext.com/ffe/siteui/vlv3/f841d4c7-10e1-40af-bcae-07a3f8dc141a/f6d7434e-d6de-4185-a6d4-c77a2d08648f/US-en-20220502-popsignuptwoweeks-perspective_alpha_website_medium.jpg');
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }

    /* TYPOGRAPHY */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        color: #E50914; /* Netflix Red */
        text-shadow: 2px 2px 4px #000000;
        font-size: 3rem !important;
        text-align: center;
        padding-bottom: 20px;
    }
    h3 {
        color: #e5e5e5;
        font-weight: 300;
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.9);
        border-right: 1px solid #333;
    }
    
    /* MOVIE CARDS (HOVER EFFECT) */
    div[data-testid="stImage"] img {
        border-radius: 10px;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.05);
        cursor: pointer;
        border: 2px solid #E50914;
    }

    /* BUTTON STYLING */
    .stButton>button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        width: 100%;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background-color: #b20710;
        color: #ffffff;
    }

    /* RATING BADGE */
    .rating-badge {
        background-color: #f5c518;
        color: black;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
    }
    
    /* LINK BUTTON (TRAILER) */
    a {
        text-decoration: none !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🧠 3. SMART DATA ENGINE
# =========================================================
@st.cache_resource
def load_and_process_data():
    if not os.path.exists('similarity.pkl'):
        # Generate data if missing (Cloud Deployment Safe)
        movies = pd.read_csv('data/tmdb_5000_movies.csv')
        credits = pd.read_csv('data/tmdb_5000_credits.csv')
        movies = movies.merge(credits, on='title')
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        movies.dropna(inplace=True)

        def convert(obj): return [i['name'] for i in ast.literal_eval(obj)]
        def convert3(obj): return [i['name'] for i in ast.literal_eval(obj)][:3]
        def fetch_director(obj): return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']

        for col in ['genres', 'keywords']: movies[col] = movies[col].apply(convert)
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
        new_df = pickle.load(open('movies.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return new_df, similarity

movies, similarity = load_and_process_data()

# =========================================================
# 🌐 4. API FUNCTIONS
# =========================================================
def fetch_details(movie_id):
    try:
        # Get Details
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url).json()
        poster = "https://image.tmdb.org/t/p/w500/" + data.get('poster_path', '') if data.get('poster_path') else "https://via.placeholder.com/500x750"
        rating = data.get('vote_average', 0)
        overview = data.get('overview', "No overview available.")
        
        # Get Trailer
        video_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
        video_data = requests.get(video_url).json()
        trailer = "None"
        if 'results' in video_data:
            for video in video_data['results']:
                if video['site'] == "YouTube" and video['type'] == "Trailer":
                    trailer = f"https://www.youtube.com/watch?v={video['key']}"
                    break
        
        return poster, rating, trailer, overview
    except:
        return "https://via.placeholder.com/500x750", 0, "None", ""

def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        results = []
        for i in movies_list:
            movie_id = movies.iloc[i[0]].movie_id
            title = movies.iloc[i[0]].title
            poster, rating, trailer, overview = fetch_details(movie_id)
            results.append((title, poster, rating, trailer, overview))
        return results
    except:
        return []

# =========================================================
# 🎬 5. APP LAYOUT
# =========================================================

# --- Header ---
st.title("CineMatch Pro")
st.markdown("<p style='text-align: center; color: #b3b3b3;'>AI-Powered Recommendations based on Content Similarity</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=50)
    st.header("Search Parameters")
    
    selected_movie = st.selectbox(
        "Select a movie you enjoyed:",
        movies['title'].values
    )
    
    if st.button('🚀 Find Recommendations'):
        search_clicked = True
    else:
        search_clicked = False
    
    st.markdown("---")
    st.markdown("© 2025 CineMatch AI")

# --- Main Content ---
if search_clicked:
    with st.spinner('Analyzing plot, genres, and cast...'):
        recommendations = recommend(selected_movie)
    
    if recommendations:
        st.subheader(f"Because you watched: {selected_movie}")
        st.markdown("") # Spacing
        
        # Grid Layout
        cols = st.columns(5)
        
        for idx, col in enumerate(cols):
            title, poster, rating, trailer, overview = recommendations[idx]
            
            with col:
                # Poster Image
                st.image(poster, use_container_width=True)
                
                # Title
                st.markdown(f"**{title}**")
                
                # Rating Badge
                st.markdown(f"<span class='rating-badge'>⭐ {round(rating, 1)}</span>", unsafe_allow_html=True)
                
                # Expander for Plot
                with st.expander("📝 Plot"):
                    st.caption(overview[:150] + "...")
                
                # Trailer Button
                if trailer != "None":
                    st.link_button("▶ Watch Trailer", trailer)
                else:
                    st.button("No Trailer", disabled=True, key=f"btn_{idx}")
    else:
        st.error("Movie not found in database! Try another one.")

else:
    # Empty State / Landing Page feel
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.info("👈 Please select a movie from the sidebar to start your discovery journey.")