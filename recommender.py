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
# ⚙️ 1. CONFIGURATION
# =========================================================
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "1d3e98627e79321f7093a1b46fe360d7")
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
# 🎨 2. PROFESSIONAL GLASSMORPHISM CSS
# =========================================================
st.markdown("""
<style>
    /* 1. APP BACKGROUND & TYPOGRAPHY */
    .stApp {
        background: linear-gradient(135deg, #090a0f 0%, #12151e 50%, #08080c 100%);
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* CUSTOM SCROLLBAR */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #090a0f;
    }
    ::-webkit-scrollbar-thumb {
        background: #2d3748;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #E50914;
    }

    /* 2. GLASSMORPHISM SIDEBAR */
    section[data-testid="stSidebar"] {
        background: rgba(8, 8, 12, 0.92);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    div[role="radiogroup"] > label > div:first-of-type {
        display: none;
    }
    div[role="radiogroup"] label {
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 6px;
        border: 1px solid transparent;
        transition: all 0.25s ease;
        cursor: pointer;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.02);
    }
    div[role="radiogroup"] label:hover {
        background: rgba(229, 9, 20, 0.15);
        color: #E50914;
        border-color: rgba(229, 9, 20, 0.3);
        transform: translateX(4px);
    }

    /* 3. HERO & HEADERS */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        line-height: 1.1;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
    }
    .hero-tagline {
        color: #E50914;
        font-size: 1.1rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 12px;
    }
    .hero-overview {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.65;
        margin-bottom: 24px;
    }

    /* 4. MOVIE CARDS & HOVER EFFECTS */
    div[data-testid="stImage"] img {
        border-radius: 14px;
        transition: transform 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.35s ease;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    }
    div[data-testid="stImage"] img:hover {
        transform: translateY(-6px) scale(1.03);
        box-shadow: 0 16px 35px rgba(229, 9, 20, 0.45);
        cursor: pointer;
    }

    /* 5. GENRE PILLS & RATING BADGES */
    .genre-pill {
        display: inline-block;
        background: rgba(229, 9, 20, 0.15);
        color: #ff5252;
        border: 1px solid rgba(229, 9, 20, 0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .rating-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ffb703, #fb8500);
        color: #000;
        font-weight: 800;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
    }

    /* 6. BUTTONS */
    div.stButton > button {
        background: linear-gradient(135deg, #E50914 0%, #B81D24 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.55rem 1.4rem;
        transition: all 0.25s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #f40d1a 0%, #d62229 100%);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.6);
        transform: translateY(-2px);
    }
    
    /* 7. FOOTER */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(5, 5, 8, 0.95);
        backdrop-filter: blur(10px);
        color: #64748b;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        z-index: 999;
    }
    
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
    
    # Enhanced TF-IDF preprocessing fallback
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
            
            rating_html = f'<span class="rating-badge">⭐ {round(movie.get("rating", 0), 1)} / 10</span>'
            st.markdown(f"{rating_html} &nbsp; • &nbsp; **{movie.get('year')}** &nbsp; • &nbsp; ⏱️ {movie.get('runtime', 'N/A')} &nbsp; • &nbsp; 🎥 **Director:** {movie.get('director', 'N/A')}", unsafe_allow_html=True)
            
            if movie.get('genres'):
                pills_html = "".join([f'<span class="genre-pill">{g}</span>' for g in movie['genres']])
                st.markdown(f'<div style="margin-top: 10px; margin-bottom: 14px;">{pills_html}</div>', unsafe_allow_html=True)
                
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
            
        # Rating Check
        r = m.get('rating', 0.0) or m.get('vote_average', 0.0) or 0.0
        if r < min_rating:
            continue
            
        # Year Range Check
        if year_range:
            try:
                raw_y = m.get('year') or m.get('release_date', '')
                y = int(str(raw_y)[:4])
                if y > 0 and (y < year_range[0] or y > year_range[1]):
                    continue
            except (ValueError, TypeError):
                pass
                
        # Genre Filter Check
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
    
    # Take top 60 candidates to allow filtering
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
    
    # --- ADVANCED FILTERS CONTROL ---
    def reset_filters():
        st.session_state["filter_genres"] = []
        st.session_state["filter_min_rating"] = 0.0
        st.session_state["filter_year_range"] = (1950, 2026)

    with st.expander("🔍 **Filter Engine**", expanded=False):
        sel_genres = st.multiselect("Genres", ALL_GENRES, key="filter_genres")
        sel_rating = st.slider("Min Rating (⭐)", 0.0, 10.0, 0.0, 0.5, key="filter_min_rating")
        sel_years = st.slider("Release Years", 1950, 2026, (1950, 2026), key="filter_year_range")
        
        st.button("🔄 Reset Filters", use_container_width=True, on_click=reset_filters)
            
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
            st.subheader(f"Because you watched '{selected}'{filter_suffix}:")
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
            st.warning("No matching movies found for the active filter settings. Try lowering the minimum rating or clearing genre filters in the sidebar!")

# --- PAGE: FAVORITES ---
elif page == "⭐ Favorites":
    st.title("⭐ My Watchlist")
    raw_watchlist = get_watchlist()
    
    f_genres = st.session_state.get("filter_genres", [])
    f_rating = st.session_state.get("filter_min_rating", 0.0)
    f_years = st.session_state.get("filter_year_range", (1950, 2026))
    
    watchlist = apply_filters(raw_watchlist, filter_genres=f_genres, min_rating=f_rating, year_range=f_years)
    
    if not raw_watchlist:
        st.info("Your watchlist is currently empty! Explore 🔥 Trending or 🎯 Recommendations to save movies you want to watch.")
    elif not watchlist:
        st.warning("No saved movies match your active filter settings in the sidebar.")
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
