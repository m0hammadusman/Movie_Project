import streamlit as st
import pandas as pd
import ast
import requests
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# ⚙️ 1. CONFIGURATION
# =========================================================
TMDB_API_KEY = "1d3e98627e79321f7093a1b46fe360d7"
DATA_DIR = "data"
MOVIES_PICKLE = "movies.pkl"
SIMILARITY_PICKLE = "similarity.pkl"

st.set_page_config(
    page_title="CineMatch Pro", 
    page_icon="🍿", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 2. PROFESSIONAL CSS (HERO & SIDEBAR)
# =========================================================
st.markdown("""
<style>
    /* 1. APP BACKGROUND */
    .stApp {
        background-color: #0f0f0f;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* 2. CUSTOM SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #222;
    }
    
    /* Hide default radio buttons circle */
    div[role="radiogroup"] > label > div:first-of-type {
        display: none;
    }
    /* Style the radio labels to look like menu items */
    div[role="radiogroup"] label {
        padding: 12px 20px;
        border-radius: 8px;
        margin-bottom: 5px;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
        font-weight: 500;
    }
    /* Hover state for menu items */
    div[role="radiogroup"] label:hover {
        background-color: #1a1a1a;
        color: #E50914;
        border-color: #333;
    }
    /* Active state is handled by Streamlit internally, but we can style the text */
    div[role="radiogroup"] label[data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }

    /* 3. HERO SECTION TYPOGRAPHY */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1.1;
        background: linear-gradient(90deg, #ffffff, #aaaaaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero-tagline {
        color: #E50914;
        font-size: 1.2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 15px;
    }
    .hero-overview {
        color: #ccc;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 25px;
    }

    /* 4. MOVIE CARDS (Grid) */
    div[data-testid="stImage"] img {
        border-radius: 12px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 30px rgba(229, 9, 20, 0.4);
        cursor: pointer;
    }

    /* 5. BUTTONS */
    div.stButton > button {
        background: #E50914;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
    }
    div.stButton > button:hover {
        background: #b20710;
        box-shadow: 0 0 10px rgba(229, 9, 20, 0.5);
    }
    
    /* 6. FOOTER */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #000;
        color: #555;
        text-align: center;
        padding: 8px;
        font-size: 11px;
        border-top: 1px solid #222;
        z-index: 999;
    }
    
    /* Hide Default Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🧠 3. DATA ENGINE
# =========================================================
@st.cache_resource
def load_data():
    # Load pickles if available
    if os.path.exists(MOVIES_PICKLE) and os.path.exists(SIMILARITY_PICKLE):
        try:
            return pickle.load(open(MOVIES_PICKLE, "rb")), pickle.load(open(SIMILARITY_PICKLE, "rb"))
        except: pass

    # Fallback: Load from CSV and process
    if not os.path.exists(os.path.join(DATA_DIR, "tmdb_5000_movies.csv")): return pd.DataFrame(), None

    movies = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_movies.csv"))
    credits = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_credits.csv"))
    movies = movies.merge(credits, on='title', how='left')
    
    # Simple preprocessing
    movies['tags'] = movies['overview'].fillna('') + " " + movies['genres'].fillna('')
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())
    
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity

movies_df, similarity = load_data()
title_to_index = {title: idx for idx, title in enumerate(movies_df['title'].values)} if not movies_df.empty else {}

# =========================================================
# 🌐 4. API FUNCTIONS
# =========================================================
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP = "https://image.tmdb.org/t/p/original"

def tmdb_get(path, params=None):
    if params is None: params = {}
    params['api_key'] = TMDB_API_KEY
    try:
        r = requests.get(f"https://api.themoviedb.org/3{path}", params=params, timeout=5)
        return r.json() if r.status_code == 200 else {}
    except: return {}

def fetch_details(movie_id):
    d = tmdb_get(f"/movie/{movie_id}", {"append_to_response": "videos,credits"})
    if not d: return {}
    
    trailer = next((f"https://youtube.com/watch?v={v['key']}" for v in d.get('videos', {}).get('results', []) if v['type'] == "Trailer"), None)
    
    return {
        "title": d.get('title'),
        "poster": TMDB_IMG + d.get('poster_path') if d.get('poster_path') else None,
        "backdrop": TMDB_BACKDROP + d.get('backdrop_path') if d.get('backdrop_path') else None,
        "overview": d.get('overview', ''),
        "rating": d.get('vote_average', 0),
        "year": d.get('release_date', 'N/A')[:4],
        "genres": [g['name'] for g in d.get('genres', [])][:3],
        "trailer": trailer,
        "id": d.get('id')
    }

def fetch_trending():
    res = tmdb_get("/trending/movie/week")
    return res.get("results", [])

def recommend(title):
    if title not in title_to_index or similarity is None: return []
    idx = title_to_index[title]
    scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:11]
    return [fetch_details(movies_df.iloc[i].movie_id) for i, _ in scores]

# =========================================================
# 🚀 5. APP LAYOUT
# =========================================================

# --- CUSTOM SIDEBAR ---
with st.sidebar:
    # Logo Area
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color:#E50914; margin:0; font-size: 28px;">CINEMATCH</h1>
        <p style="color:#666; font-size: 12px; margin-top:5px;">AI POWERED DISCOVERY</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Menu (Radio button styled as Menu)
    page = st.radio("MENU", ["🔥 Trending", "🎯 Recommendations", "⭐ Favorites"], label_visibility="collapsed")
    
    st.markdown("---")
    st.caption("Data provided by TMDB")

# --- PAGE: TRENDING (HERO SECTION) ---
if page == "🔥 Trending":
    trending = fetch_trending()
    
    if trending:
        top_movie = trending[0]
        hero = fetch_details(top_movie['id'])
        
        # --- HERO SECTION START ---
        # We use columns to create a "Split" Hero: Text on Left, Big Image on Right
        hero_container = st.container()
        with hero_container:
            col1, col2 = st.columns([1, 1.5], gap="large")
            
            with col1:
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True) # Spacer
                st.markdown('<p class="hero-tagline">#1 Trending Movie</p>', unsafe_allow_html=True)
                st.markdown(f'<h1 class="hero-title">{hero["title"]}</h1>', unsafe_allow_html=True)
                
                # Metadata tags
                st.markdown(f"**{hero['year']}** • ⭐ **{round(hero['rating'], 1)}** • _{', '.join(hero['genres'])}_")
                
                st.markdown(f'<p class="hero-overview">{hero["overview"][:200]}...</p>', unsafe_allow_html=True)
                
                # Buttons Row
                b1, b2 = st.columns([1, 2])
                with b1:
                    if hero['trailer']:
                        st.link_button("▶ Play Trailer", hero['trailer'])
                    else:
                        st.button("No Trailer", disabled=True)
                with b2:
                    if st.button("More Info"):
                        st.session_state.selected_movie = hero['title']
                        st.toast(f"Selected {hero['title']}")

            with col2:
                # The Backdrop Image with Shadow
                if hero['backdrop']:
                    st.image(hero['backdrop'], use_container_width=True)
        # --- HERO SECTION END ---

        st.markdown("---")
        
        # Grid Section for other trending
        st.subheader("Top Picks This Week")
        for i in range(1, 11, 5): # Skip #1
            cols = st.columns(5)
            for j in range(5):
                if i + j < len(trending):
                    m = trending[i + j]
                    with cols[j]:
                        poster = TMDB_IMG + m.get('poster_path', '')
                        st.image(poster, use_container_width=True)
                        st.write(f"**{m['title']}**")
                        st.caption(f"⭐ {round(m['vote_average'], 1)}")

# --- PAGE: RECOMMENDATIONS ---
elif page == "🎯 Recommendations":
    st.title("Find Your Next Obsession")
    
    # Search Bar Container
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            options = movies_df['title'].values if not movies_df.empty else []
            selected = st.selectbox("I enjoyed watching...", options)
        with c2:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True) # Align button
            if st.button("Get Recommendations", type="primary", use_container_width=True):
                st.session_state.trigger_rec = True

    if st.session_state.get("trigger_rec"):
        recs = recommend(selected)
        if recs:
            st.subheader(f"Because you watched '{selected}':")
            st.markdown("")
            
            # Responsive Grid
            for i in range(0, len(recs), 5):
                cols = st.columns(5)
                for j in range(5):
                    if i + j < len(recs):
                        m = recs[i + j]
                        with cols[j]:
                            st.image(m['poster'], use_container_width=True)
                            st.markdown(f"**{m['title']}**")
                            st.caption(f"{m['year']} • ⭐ {round(m['rating'], 1)}")
                            if m['trailer']:
                                st.link_button("Trailer", m['trailer'])
        else:
            st.warning("No matches found in database.")

# --- PAGE: FAVORITES (Placeholder) ---
elif page == "⭐ Favorites":
    st.title("My List")
    st.info("This feature is coming in the next update! (Requires Database)")

# --- FOOTER ---
st.markdown('<div class="footer">Designed by <b>Usman</b> • Powered by TMDB API</div>', unsafe_allow_html=True)