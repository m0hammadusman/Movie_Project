import streamlit as st
import pandas as pd
import ast
import requests
import os
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

USER_DATA_FILE = "user_data.json"

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"profiles": {"default": {"watchlist": [], "ratings": {}, "prefs": {"genres": [], "actors": []}, "history": []}}, "active_profile": "default"}

def save_user_data(data):
    try:
        with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving user data: {e}")

def get_watchlist():
    data = load_user_data()
    active = data.get("active_profile", "default")
    return data.get("profiles", {}).get(active, {}).get("watchlist", [])

def is_in_watchlist(movie_id, title=""):
    watchlist = get_watchlist()
    for item in watchlist:
        if (movie_id and item.get("id") == movie_id) or (title and item.get("title") == title):
            return True
    return False

def toggle_watchlist(movie_dict):
    data = load_user_data()
    active = data.get("active_profile", "default")
    if "profiles" not in data:
        data["profiles"] = {}
    if active not in data["profiles"]:
        data["profiles"][active] = {"watchlist": [], "ratings": {}, "prefs": {"genres": [], "actors": []}, "history": []}
    watchlist = data["profiles"][active]["watchlist"]
    
    m_id = movie_dict.get("id")
    m_title = movie_dict.get("title")
    
    existing_idx = None
    for idx, item in enumerate(watchlist):
        if (m_id and item.get("id") == m_id) or (m_title and item.get("title") == m_title):
            existing_idx = idx
            break
            
    if existing_idx is not None:
        watchlist.pop(existing_idx)
        added = False
    else:
        watchlist.append({
            "id": m_id,
            "title": m_title,
            "poster": movie_dict.get("poster"),
            "year": movie_dict.get("year", "N/A"),
            "rating": movie_dict.get("rating", 0),
            "genres": movie_dict.get("genres", []),
            "overview": movie_dict.get("overview", ""),
            "trailer": movie_dict.get("trailer")
        })
        added = True
    save_user_data(data)
    return added


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
    if not trailer:
        trailer = next((f"https://youtube.com/watch?v={v['key']}" for v in d.get('videos', {}).get('results', [])), None)
        
    cast_list = [c['name'] for c in d.get('credits', {}).get('cast', [])[:5]]
    director = next((c['name'] for c in d.get('credits', {}).get('crew', []) if c.get('job') == 'Director'), 'N/A')
    
    return {
        "title": d.get('title'),
        "poster": TMDB_IMG + d.get('poster_path') if d.get('poster_path') else None,
        "backdrop": TMDB_BACKDROP + d.get('backdrop_path') if d.get('backdrop_path') else None,
        "overview": d.get('overview', ''),
        "rating": d.get('vote_average', 0),
        "year": d.get('release_date', 'N/A')[:4] if d.get('release_date') else "N/A",
        "genres": [g['name'] for g in d.get('genres', [])][:4],
        "trailer": trailer,
        "id": d.get('id'),
        "tagline": d.get('tagline', ''),
        "runtime": f"{d.get('runtime')} mins" if d.get('runtime') else "N/A",
        "cast": cast_list,
        "director": director,
        "vote_count": d.get('vote_count', 0)
    }

def render_movie_modal(movie):
    if not movie:
        return
        
    st.markdown("---")
    with st.container():
        m_col1, m_col2 = st.columns([1, 1.5], gap="large")
        with m_col1:
            if movie.get('poster'):
                st.image(movie['poster'], use_container_width=True)
            elif movie.get('backdrop'):
                st.image(movie['backdrop'], use_container_width=True)
        with m_col2:
            st.markdown(f"## 🎬 {movie.get('title')}")
            if movie.get('tagline'):
                st.markdown(f"*\"{movie['tagline']}\"*")
            
            st.markdown(f"**Year:** {movie.get('year')}  |  **Rating:** ⭐ **{round(movie.get('rating', 0), 1)}/10** ({movie.get('vote_count', 0):,} votes)  |  **Runtime:** ⏱️ {movie.get('runtime', 'N/A')}")
            st.markdown(f"**Director:** 🎥 {movie.get('director', 'N/A')}")
            if movie.get('genres'):
                st.markdown(f"**Genres:** {', '.join(movie['genres'])}")
            if movie.get('cast'):
                st.markdown(f"**Starring:** {', '.join(movie['cast'])}")
                
            st.markdown("### 📝 Synopsis")
            st.markdown(movie.get('overview', 'No overview available.'))
            
            btn_c1, btn_c2 = st.columns([1, 1])
            with btn_c1:
                is_fav = is_in_watchlist(movie.get('id'), movie.get('title'))
                btn_label = "❤️ In Watchlist" if is_fav else "+ Add to Watchlist"
                if st.button(btn_label, key=f"modal_fav_{movie.get('id') or movie.get('title')}"):
                    added = toggle_watchlist(movie)
                    st.toast("Added to Watchlist!" if added else "Removed from Watchlist")
                    st.rerun()
            with btn_c2:
                if st.button("❌ Close Details", key="close_modal_btn"):
                    del st.session_state["active_movie_modal"]
                    st.rerun()
                    
        # Embedded Video Player if trailer exists
        if movie.get('trailer'):
            st.markdown("### 🍿 Official Trailer")
            st.video(movie['trailer'])
    st.markdown("---")


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

# --- ACTIVE MOVIE DETAIL MODAL HANDLER ---
if "active_movie_modal" in st.session_state and st.session_state.active_movie_modal:
    modal_target = st.session_state.active_movie_modal
    modal_details = None
    if isinstance(modal_target, int) or (isinstance(modal_target, str) and modal_target.isdigit()):
        modal_details = fetch_details(int(modal_target))
    elif isinstance(modal_target, dict):
        modal_details = fetch_details(modal_target.get('id')) if modal_target.get('id') else modal_target
    elif isinstance(modal_target, str):
        t_match = movies_df[movies_df['title'] == modal_target] if not movies_df.empty else pd.DataFrame()
        if not t_match.empty:
            modal_details = fetch_details(t_match.iloc[0].movie_id)
        else:
            modal_details = {"title": modal_target}
    if modal_details:
        render_movie_modal(modal_details)

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
                b1, b2, b3 = st.columns([1.2, 1.2, 1])
                with b1:
                    if hero['trailer']:
                        st.link_button("▶ Play Trailer", hero['trailer'])
                    else:
                        st.button("No Trailer", disabled=True)
                with b2:
                    is_fav = is_in_watchlist(hero.get('id'), hero.get('title'))
                    fav_btn_label = "❤️ In Watchlist" if is_fav else "+ Watchlist"
                    if st.button(fav_btn_label, key=f"fav_hero_{hero.get('id')}"):
                        added = toggle_watchlist(hero)
                        st.toast("Added to Watchlist!" if added else "Removed from Watchlist")
                        st.rerun()
                with b3:
                    if st.button("ℹ️ More Info", key=f"info_hero_{hero.get('id')}"):
                        st.session_state.active_movie_modal = hero.get('id')
                        st.rerun()

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
                        
                        m_obj = {
                            "id": m.get("id"),
                            "title": m.get("title"),
                            "poster": poster,
                            "year": m.get("release_date", "N/A")[:4] if m.get("release_date") else "N/A",
                            "rating": m.get("vote_average", 0),
                            "overview": m.get("overview", "")
                        }
                        tc1, tc2 = st.columns(2)
                        with tc1:
                            is_fav = is_in_watchlist(m.get('id'), m.get('title'))
                            icon = "❤️ Saved" if is_fav else "+ Save"
                            if st.button(icon, key=f"fav_trend_{i}_{j}_{m.get('id')}"):
                                added = toggle_watchlist(m_obj)
                                st.toast(f"{'Saved' if added else 'Removed'} '{m['title']}'")
                                st.rerun()
                        with tc2:
                            if st.button("ℹ️ Info", key=f"info_trend_{i}_{j}_{m.get('id')}"):
                                st.session_state.active_movie_modal = m.get('id')
                                st.rerun()

# --- PAGE: RECOMMENDATIONS ---
elif page == "🎯 Recommendations":
    st.title("Find Your Next Obsession")
    
    # Search Bar Container
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            options = list(movies_df['title'].values) if not movies_df.empty else []
            default_index = 0
            if "fav_selected_movie" in st.session_state and st.session_state.fav_selected_movie in options:
                default_index = options.index(st.session_state.fav_selected_movie)
            selected = st.selectbox("I enjoyed watching...", options, index=default_index)
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
                            if m.get('poster'):
                                st.image(m['poster'], use_container_width=True)
                            st.markdown(f"**{m['title']}**")
                            st.caption(f"{m['year']} • ⭐ {round(m['rating'], 1)}")
                            rc1, rc2, rc3 = st.columns([1.1, 1, 1])
                            with rc1:
                                if m.get('trailer'):
                                    st.link_button("▶ Trailer", m['trailer'])
                            with rc2:
                                is_fav = is_in_watchlist(m.get('id'), m.get('title'))
                                fav_icon = "❤️ Saved" if is_fav else "+ Save"
                                if st.button(fav_icon, key=f"fav_rec_{i}_{j}_{m.get('id') or m.get('title')}"):
                                    added = toggle_watchlist(m)
                                    st.toast(f"{'Saved' if added else 'Removed'} '{m['title']}'")
                                    st.rerun()
                            with rc3:
                                if st.button("ℹ️ Info", key=f"info_rec_{i}_{j}_{m.get('id') or m.get('title')}"):
                                    st.session_state.active_movie_modal = m.get('id') or m.get('title')
                                    st.rerun()
        else:
            st.warning("No matches found in database.")

# --- PAGE: FAVORITES ---
elif page == "⭐ Favorites":
    st.title("⭐ My Watchlist")
    watchlist = get_watchlist()
    
    if not watchlist:
        st.info("Your watchlist is currently empty! Explore 🔥 Trending or 🎯 Recommendations to save movies you want to watch.")
    else:
        st.markdown(f"**{len(watchlist)} Saved Movie{'s' if len(watchlist) != 1 else ''}** in your collection")
        st.markdown("---")
        
        # Responsive Grid
        for i in range(0, len(watchlist), 5):
            cols = st.columns(5)
            for j in range(5):
                if i + j < len(watchlist):
                    m = watchlist[i + j]
                    with cols[j]:
                        if m.get('poster'):
                            st.image(m['poster'], use_container_width=True)
                        st.markdown(f"**{m.get('title')}**")
                        st.caption(f"{m.get('year', 'N/A')} • ⭐ {round(m.get('rating', 0), 1)}")
                        
                        fc1, fc2, fc3 = st.columns([1, 1, 1])
                        with fc1:
                            if st.button("🎯 Recs", key=f"rec_fav_{i}_{j}_{m.get('id') or m.get('title')}"):
                                st.session_state.fav_selected_movie = m.get('title')
                                st.session_state.trigger_rec = True
                                st.toast(f"Finding recommendations for '{m.get('title')}'...")
                                st.rerun()
                        with fc2:
                            if st.button("ℹ️ Info", key=f"info_fav_{i}_{j}_{m.get('id') or m.get('title')}"):
                                st.session_state.active_movie_modal = m.get('id') or m.get('title')
                                st.rerun()
                        with fc3:
                            if st.button("🗑️", key=f"del_fav_{i}_{j}_{m.get('id') or m.get('title')}"):
                                toggle_watchlist(m)
                                st.toast(f"Removed '{m.get('title')}' from watchlist")
                                st.rerun()

# --- FOOTER ---
st.markdown('<div class="footer">Designed by <b>Usman</b> • Powered by TMDB API</div>', unsafe_allow_html=True)
