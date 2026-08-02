import streamlit as st
import pandas as pd
import ast
import requests
import os
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
# ⚙️ CONFIGURATION
# =========================================================
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "1d3e98627e79321f7093a1b46fe360d7")
DATA_DIR = "data"
MOVIES_PICKLE = "movies.pkl"
SIMILARITY_PICKLE = "similarity.pkl"

st.set_page_config(
    page_title="CINEMATCH — AI Movie Discovery", 
    page_icon="🍿", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 EXACT USER-PROVIDED HTML & CSS TEMPLATE STYLES
# =========================================================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

<style>
    /* --- RESET & GLOBALS --- */
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    .stDeployButton { display: none !important; }

    html, body, .stApp {
      background-color: #0b0d13 !important;
      color: #ffffff;
    }

    .main .block-container {
      padding-top: 0rem !important;
      padding-bottom: 2rem !important;
      padding-left: 0rem !important;
      padding-right: 0rem !important;
      max-width: 100% !important;
    }

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {
      width: 260px !important;
      background-color: #0f1118 !important;
      border-right: 1px solid #1a1d28 !important;
      padding: 24px 20px !important;
    }

    .logo-title {
      color: #e50914;
      font-size: 20px;
      font-weight: 900;
      letter-spacing: 1px;
    }

    .logo-subtitle {
      color: #5d6578;
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 1px;
      margin-top: 2px;
    }

    .nav-menu {
      margin-top: 32px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    /* Streamlit Sidebar Buttons override */
    div[data-testid="stSidebar"] div.stButton > button {
      display: flex !important;
      align-items: center !important;
      justify-content: flex-start !important;
      gap: 12px !important;
      padding: 12px 16px !important;
      color: #8c94a8 !important;
      background-color: transparent !important;
      font-size: 14px !important;
      font-weight: 500 !important;
      border-radius: 8px !important;
      border: none !important;
      width: 100% !important;
      text-align: left !important;
      transition: all 0.2s !important;
      margin-bottom: 2px !important;
    }

    div[data-testid="stSidebar"] div.stButton > button:hover {
      color: #fff !important;
      background-color: #181b26 !important;
    }

    .nav-item-active div.stButton > button {
      background-color: #921c22 !important;
      color: #ffffff !important;
      font-weight: 600 !important;
    }

    .quick-filters {
      margin-top: 36px;
    }

    .filter-title {
      color: #ffffff;
      font-size: 13px;
      font-weight: 700;
      display: block;
      margin-bottom: 12px;
    }

    .filter-chip-btn div.stButton > button {
      background-color: #1a1d27 !important;
      color: #a0a8be !important;
      border: none !important;
      padding: 6px 12px !important;
      border-radius: 16px !important;
      font-size: 12px !important;
      cursor: pointer !important;
      justify-content: center !important;
    }

    .filter-chip-btn div.stButton > button:hover {
      background-color: #262a38 !important;
      color: #fff !important;
    }

    .sidebar-footer {
      font-size: 12px;
      color: #5d6578;
      border-top: 1px solid #1a1d28;
      padding-top: 16px;
      margin-top: 40px;
    }

    .status-indicator {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-top: 6px;
      color: #a0a8be;
    }

    .dot {
      width: 6px;
      height: 6px;
      background-color: #22c55e;
      border-radius: 50%;
    }

    /* Top Header */
    .top-header-wrap {
      padding: 16px 36px;
    }

    .user-profile {
      display: flex;
      align-items: center;
      gap: 12px;
      float: right;
    }

    .avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      object-fit: cover;
    }

    .user-info {
      display: flex;
      flex-direction: column;
    }

    .user-name {
      font-size: 13px;
      font-weight: 600;
      color: #fff;
    }

    .user-badge {
      font-size: 10px;
      color: #5d6578;
    }

    /* Hero Banner */
    .hero-section-wrap {
      position: relative;
      min-height: 380px;
      padding: 30px 36px;
    }

    .trending-badge {
      background-color: #1a1d27;
      color: #fff;
      font-size: 10px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }

    .movie-title {
      font-size: 42px;
      font-weight: 800;
      line-height: 1.1;
      margin: 12px 0 10px 0;
      color: #fff;
    }

    .gradient-text {
      background: linear-gradient(90deg, #d8809a 0%, #a291f0 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .movie-meta {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: #a0a8be;
      margin-bottom: 12px;
    }

    .rating {
      color: #facc15;
      font-weight: 700;
    }

    .pg-badge {
      border: 1px solid #3a3f54;
      padding: 1px 4px;
      border-radius: 3px;
      font-size: 10px;
    }

    .movie-synopsis {
      font-size: 13px;
      color: #8c94a8;
      line-height: 1.5;
      margin-bottom: 20px;
      max-width: 540px;
    }

    /* Main Body Buttons */
    .btn-primary-red div.stButton > button {
      background-color: #e50914 !important;
      color: #fff !important;
      padding: 10px 20px !important;
      border-radius: 6px !important;
      font-size: 13px !important;
      font-weight: 600 !important;
      border: none !important;
    }

    .btn-secondary-dark div.stButton > button {
      background-color: #202433 !important;
      color: #fff !important;
      padding: 10px 20px !important;
      border-radius: 6px !important;
      font-size: 13px !important;
      font-weight: 600 !important;
      border: none !important;
    }

    /* Recommendations Section */
    .recommendations-section-wrap {
      padding: 24px 36px;
    }

    .section-title {
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.5px;
      margin-bottom: 16px;
      color: #a0a8be;
    }

    .card-rating {
      position: absolute;
      top: 6px;
      right: 6px;
      background-color: rgba(0, 0, 0, 0.7);
      color: #facc15;
      font-size: 10px;
      font-weight: 700;
      padding: 3px 6px;
      border-radius: 4px;
      backdrop-filter: blur(4px);
    }

    .card-title {
      font-size: 12px;
      font-weight: 600;
      color: #fff;
      margin-top: 6px;
    }

    div[data-testid="stImage"] img {
      border-radius: 8px;
      transition: transform 0.2s;
    }
    div[data-testid="stImage"] img:hover {
      transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🧠 DATA ENGINE
# =========================================================
@st.cache_resource
def load_data():
    if os.path.exists(MOVIES_PICKLE) and os.path.exists(SIMILARITY_PICKLE):
        try:
            return pickle.load(open(MOVIES_PICKLE, "rb")), pickle.load(open(SIMILARITY_PICKLE, "rb"))
        except: pass

    if not os.path.exists(os.path.join(DATA_DIR, "tmdb_5000_movies.csv")): return pd.DataFrame(), None

    movies = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_movies.csv"))
    credits = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_credits.csv"))
    movies = movies.merge(credits, on='title', how='left')
    
    movies['tags'] = movies['overview'].fillna('') + " " + movies['genres'].fillna('')
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())
    
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    vectors = tfidf.fit_transform(new_df['tags'])
    similarity = cosine_similarity(vectors)

    return new_df, similarity

movies_df, similarity = load_data()
title_to_index = {title: idx for idx, title in enumerate(movies_df['title'].values)} if not movies_df.empty else {}

# =========================================================
# 🌐 API FUNCTIONS
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
            
            rating_html = f'<span class="rating"><i class="fa-solid fa-star"></i> {round(movie.get("rating", 0), 1)} / 10</span>'
            st.markdown(f"{rating_html} &nbsp; • &nbsp; **{movie.get('year')}** &nbsp; • &nbsp; ⏱️ {movie.get('runtime', 'N/A')} &nbsp; • &nbsp; 🎥 **Director:** {movie.get('director', 'N/A')}", unsafe_allow_html=True)
            
            if movie.get('genres'):
                pills_html = "".join([f'<span style="display:inline-block; background:#1a1d27; color:#a0a8be; border-radius:16px; padding:4px 10px; font-size:12px; margin-right:6px;">{g}</span>' for g in movie['genres']])
                st.markdown(f'<div style="margin-top: 10px; margin-bottom: 14px;">{pills_html}</div>', unsafe_allow_html=True)
                
            if movie.get('cast'):
                st.markdown(f"**Starring:** {', '.join(movie['cast'])}")
                
            st.markdown("### 📝 Synopsis")
            st.markdown(movie.get('overview', 'No overview available.'))
            
            btn_c1, btn_c2 = st.columns([1, 1])
            with btn_c1:
                is_fav = is_in_watchlist(movie.get('id'), movie.get('title'))
                btn_label = "❤️ In List" if is_fav else "+ Add to List"
                st.markdown('<div class="btn-primary-red">', unsafe_allow_html=True)
                if st.button(btn_label, key=f"modal_fav_{movie.get('id') or movie.get('title')}"):
                    added = toggle_watchlist(movie)
                    st.toast("Added to List!" if added else "Removed from List")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with btn_c2:
                st.markdown('<div class="btn-secondary-dark">', unsafe_allow_html=True)
                if st.button("❌ Close", key="close_modal_btn"):
                    del st.session_state["active_movie_modal"]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                    
        if movie.get('trailer'):
            st.markdown("### 🍿 Official Trailer")
            st.video(movie['trailer'])
    st.markdown("---")

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
]

def apply_filters(movie_list, filter_genres=None, min_rating=0.0, year_range=None):
    if not movie_list:
        return []
    filtered = []
    for m in movie_list:
        if not m:
            continue
            
        r = m.get('rating', 0.0) or m.get('vote_average', 0.0) or 0.0
        if r < min_rating:
            continue
            
        if year_range:
            try:
                raw_y = m.get('year') or m.get('release_date', '')
                y = int(str(raw_y)[:4])
                if y > 0 and (y < year_range[0] or y > year_range[1]):
                    continue
            except (ValueError, TypeError):
                pass
                
        if filter_genres:
            m_genres = [g.lower() for g in m.get('genres', [])]
            if not any(fg.lower() in m_genres for fg in filter_genres):
                continue
                
        filtered.append(m)
    return filtered

def fetch_trending():
    res = tmdb_get("/trending/movie/week")
    return res.get("results", [])

def recommend(title, filter_genres=None, min_rating=0.0, year_range=None, max_results=10):
    if title not in title_to_index or similarity is None: return []
    idx = title_to_index[title]
    
    scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:60]
    
    results = []
    for i, _ in scores:
        details = fetch_details(movies_df.iloc[i].movie_id)
        if not details:
            continue
            
        r = details.get('rating', 0.0) or 0.0
        if r < min_rating:
            continue
            
        if year_range:
            try:
                y = int(str(details.get('year', '0'))[:4])
                if y > 0 and (y < year_range[0] or y > year_range[1]):
                    continue
            except (ValueError, TypeError):
                pass
                
        if filter_genres:
            m_genres = [g.lower() for g in details.get('genres', [])]
            if not any(fg.lower() in m_genres for fg in filter_genres):
                continue
                
        results.append(details)
        if len(results) >= max_results:
            break
            
    return results

# =========================================================
# 🚀 APP NAVIGATION & STATE
# =========================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Trending"

# --- LEFT SIDEBAR (EXACT USER TEMPLATE MATCH) ---
with st.sidebar:
    st.markdown("""
    <div class="logo">
      <h1 class="logo-title">CINEMATCH</h1>
      <p class="logo-subtitle">AI-POWERED DISCOVERY</p>
    </div>
    """, unsafe_allow_html=True)
    
    nav_items = [
        ("Trending", "nav_tr"),
        ("Recommendations", "nav_rec"),
        ("Favorites", "nav_fav"),
        ("Watch History", "nav_hist")
    ]
    
    st.markdown('<div class="nav-menu">', unsafe_allow_html=True)
    for label, key in nav_items:
        is_active = (st.session_state.current_page == label)
        if is_active:
            st.markdown('<div class="nav-item-active">', unsafe_allow_html=True)
            if st.button(f"{label}", key=key):
                st.session_state.current_page = label
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.button(f"{label}", key=key):
                st.session_state.current_page = label
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick Filters
    st.markdown("""
    <div class="quick-filters">
      <span class="filter-title">Quick Filters</span>
    </div>
    """, unsafe_allow_html=True)
    
    q1, q2 = st.columns(2)
    with q1:
        st.markdown('<div class="filter-chip-btn">', unsafe_allow_html=True)
        if st.button("Action", key="qf_act"):
            st.session_state.filter_genres = ["Action"]
            st.rerun()
        if st.button("Mind-Bending", key="qf_mb"):
            st.session_state.filter_genres = ["Science Fiction", "Mystery"]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="filter-chip-btn">', unsafe_allow_html=True)
        if st.button("Sci-Fi", key="qf_sf"):
            st.session_state.filter_genres = ["Science Fiction"]
            st.rerun()
        if st.button("Space", key="qf_sp"):
            st.session_state.filter_genres = ["Science Fiction", "Adventure"]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    def reset_filters():
        st.session_state["filter_genres"] = []
        st.session_state["filter_min_rating"] = 0.0
        st.session_state["filter_year_range"] = (1950, 2026)

    with st.expander("🔍 **Filter Engine**", expanded=False):
        sel_genres = st.multiselect("Genres", ALL_GENRES, key="filter_genres")
        sel_rating = st.slider("Min Rating (⭐)", 0.0, 10.0, 0.0, 0.5, key="filter_min_rating")
        sel_years = st.slider("Release Years", 1950, 2026, (1950, 2026), key="filter_year_range")
        st.button("🔄 Reset Filters", use_container_width=True, on_click=reset_filters)

    st.markdown("""
    <div class="sidebar-footer">
      <p>Data by TMDB</p>
      <div class="status-indicator">
        <span class="dot"></span> Online
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- TOP SEARCH & USER PROFILE HEADER (EXACT USER TEMPLATE MATCH) ---
top_col1, top_col2 = st.columns([3.6, 1])

with top_col1:
    st.markdown('<div class="top-header-wrap">', unsafe_allow_html=True)
    sc1, sc2 = st.columns([4.2, 1])
    with sc1:
        ai_prompt = st.text_input(
            "Ask AI Search", 
            placeholder='Ask AI: "Show me movies like interstellar with a mind-bending twist..."',
            label_visibility="collapsed"
        )
    with sc2:
        if st.button("Match", type="primary", use_container_width=True):
            if ai_prompt:
                st.session_state.fav_selected_movie = ai_prompt
                st.session_state.trigger_rec = True
                st.session_state.current_page = "Recommendations"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with top_col2:
    st.markdown("""
    <div class="top-header-wrap">
      <div class="user-profile">
        <img src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&q=80&w=100" alt="Alex Morgan" class="avatar">
        <div class="user-info">
          <span class="user-name">Alex Morgan <i class="fa-solid fa-chevron-down" style="font-size: 10px;"></i></span>
          <span class="user-badge">Pro Member</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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

# --- PAGE: TRENDING (EXACT USER HERO & RECOMMENDATIONS GRID) ---
if st.session_state.current_page in ["Trending", "Watch History"]:
    trending = fetch_trending()
    
    if trending:
        top_movie = trending[0]
        hero = fetch_details(top_movie['id'])
        
        # --- HERO BANNER ---
        hero_bg_url = hero.get('backdrop') or 'http://googleusercontent.com/image_collection/image_retrieval/6629781975895960306_0'
        
        hero_container = st.container()
        with hero_container:
            st.markdown(f'''
            <div style="position: relative; min-height: 400px; padding: 40px 36px; display: flex; align-items: flex-end; background-image: url('{hero_bg_url}'); background-size: cover; background-position: center top;">
              <div class="hero-overlay-box"></div>
            </div>
            ''', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1.1, 1], gap="large")
            with col1:
                st.markdown('<span class="trending-badge">#1 TRENDING MOVIE</span>', unsafe_allow_html=True)
                st.markdown(f'<h1 class="movie-title">{hero["title"]}</h1>', unsafe_allow_html=True)
                
                rating_val = round(hero.get('rating', 8.0), 1)
                genres_str = ", ".join(hero.get('genres', ['Sci-Fi', 'Action', 'Adventure']))
                st.markdown(f'''
                <div class="movie-meta">
                  <span class="rating"><i class="fa-solid fa-star"></i> {rating_val}</span>
                  <span>•</span>
                  <span>{hero.get("year", "2026")}</span>
                  <span>•</span>
                  <span class="pg-badge">PG-13</span>
                  <span>•</span>
                  <span class="genres">{genres_str}</span>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'<p class="movie-synopsis">{hero["overview"][:220]}...</p>', unsafe_allow_html=True)
                
                hb1, hb2, hb3 = st.columns([1.2, 1.2, 1])
                with hb1:
                    st.markdown('<div class="btn-primary-red">', unsafe_allow_html=True)
                    if hero.get('trailer'):
                        st.link_button("▶ Watch Trailer", hero['trailer'])
                    else:
                        st.button("No Trailer", disabled=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with hb2:
                    is_fav = is_in_watchlist(hero.get('id'), hero.get('title'))
                    fav_label = "❤️ Saved" if is_fav else "+ Add to List"
                    st.markdown('<div class="btn-secondary-dark">', unsafe_allow_html=True)
                    if st.button(fav_label, key=f"fav_hero_{hero.get('id')}"):
                        added = toggle_watchlist(hero)
                        st.toast("Added to List!" if added else "Removed from List")
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                with hb3:
                    st.markdown('<div class="btn-secondary-dark">', unsafe_allow_html=True)
                    if st.button("ℹ️ Info", key=f"info_hero_{hero.get('id')}"):
                        st.session_state.active_movie_modal = hero.get('id')
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                if hero.get('backdrop'):
                    st.image(hero['backdrop'], use_container_width=True)

        # --- RECOMMENDATIONS GRID SECTION ---
        st.markdown('<div class="recommendations-section-wrap">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">AI RECOMMENDED FOR YOU</h2>', unsafe_allow_html=True)
        
        for i in range(1, 11, 5): # Skip #1
            cols = st.columns(5)
            for j in range(5):
                if i + j < len(trending):
                    m = trending[i + j]
                    with cols[j]:
                        poster = TMDB_IMG + m.get('poster_path', '') if m.get('poster_path') else 'https://images.unsplash.com/photo-1534447677768-be436bb09401?auto=format&fit=crop&q=80&w=400'
                        vote_avg = round(m.get('vote_average', 8.0), 1)
                        
                        st.image(poster, use_container_width=True)
                        st.markdown(f'<h3 class="card-title">{m["title"]} &nbsp; <span class="rating"><i class="fa-solid fa-star"></i> {vote_avg}</span></h3>', unsafe_allow_html=True)
                        
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
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: RECOMMENDATIONS ---
elif st.session_state.current_page == "Recommendations":
    st.markdown('<div class="recommendations-section-wrap">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">FIND YOUR NEXT OBSESSION</h2>', unsafe_allow_html=True)
    
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            options = list(movies_df['title'].values) if not movies_df.empty else []
            default_index = 0
            if "fav_selected_movie" in st.session_state and st.session_state.fav_selected_movie in options:
                default_index = options.index(st.session_state.fav_selected_movie)
            selected = st.selectbox("I enjoyed watching...", options, index=default_index)
        with c2:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            if st.button("Get Recommendations", type="primary", use_container_width=True):
                st.session_state.trigger_rec = True

    f_genres = st.session_state.get("filter_genres", [])
    f_rating = st.session_state.get("filter_min_rating", 0.0)
    f_years = st.session_state.get("filter_year_range", (1950, 2026))

    if st.session_state.get("trigger_rec"):
        recs = recommend(selected, filter_genres=f_genres, min_rating=f_rating, year_range=f_years)
        if recs:
            filter_labels = []
            if f_genres: filter_labels.append(f"Genres: {', '.join(f_genres)}")
            if f_rating > 0: filter_labels.append(f"Rating ≥ ⭐ {f_rating}")
            if f_years != (1950, 2026): filter_labels.append(f"Years: {f_years[0]}-{f_years[1]}")
            
            filter_suffix = f" • Filters Applied: [{', '.join(filter_labels)}]" if filter_labels else ""
            st.markdown(f'<h2 class="section-title">BECAUSE YOU WATCHED \'{selected}\'{filter_suffix}:</h2>', unsafe_allow_html=True)
            
            for i in range(0, len(recs), 5):
                cols = st.columns(5)
                for j in range(5):
                    if i + j < len(recs):
                        m = recs[i + j]
                        with cols[j]:
                            if m.get('poster'):
                                st.image(m['poster'], use_container_width=True)
                            st.markdown(f'<h3 class="card-title">{m["title"]}</h3>', unsafe_allow_html=True)
                            
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
            st.warning("No matching movies found for the active filter settings.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: FAVORITES ---
elif st.session_state.current_page == "Favorites":
    st.markdown('<div class="recommendations-section-wrap">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">⭐ MY WATCHLIST COLLECTION</h2>', unsafe_allow_html=True)
    raw_watchlist = get_watchlist()
    
    f_genres = st.session_state.get("filter_genres", [])
    f_rating = st.session_state.get("filter_min_rating", 0.0)
    f_years = st.session_state.get("filter_year_range", (1950, 2026))
    
    watchlist = apply_filters(raw_watchlist, filter_genres=f_genres, min_rating=f_rating, year_range=f_years)
    
    if not raw_watchlist:
        st.info("Your watchlist is currently empty! Explore Trending or Recommendations to save movies.")
    elif not watchlist:
        st.warning("No saved movies match your active filter settings.")
    else:
        st.markdown(f"**{len(watchlist)} Saved Movie{'s' if len(watchlist) != 1 else ''}** in your collection")
        st.markdown("---")
        
        for i in range(0, len(watchlist), 5):
            cols = st.columns(5)
            for j in range(5):
                if i + j < len(watchlist):
                    m = watchlist[i + j]
                    with cols[j]:
                        if m.get('poster'):
                            st.image(m['poster'], use_container_width=True)
                        st.markdown(f'<h3 class="card-title">{m.get("title")}</h3>', unsafe_allow_html=True)
                        
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
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center; padding:15px; color:#5d6578; font-size:11px;">CINEMATCH • Data by TMDB • Designed by Usman</div>', unsafe_allow_html=True)
