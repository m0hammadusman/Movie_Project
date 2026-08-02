# 🍿 CineMatch Pro — AI-Powered Movie Discovery & Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF-orange.svg)
![Deployment](https://img.shields.io/badge/Streamlit%20Cloud-100%25%20Free-brightgreen.svg)
![TMDB API](https://img.shields.io/badge/TMDB-API%20v3-green.svg)

**CineMatch Pro** is a modern, high-performance web application for discovering movies, getting AI-driven content recommendations, viewing embedded trailers, filtering by genres/ratings, and managing a personal watchlist.

---

## 🌐 Free Live Deployment Guide (Streamlit Community Cloud)

This app is optimized for **100% Free Lifetime Hosting** on **Streamlit Community Cloud**:

1. Push this repository to your **GitHub** account.
2. Visit **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New App"** and fill in:
   - **Repository**: `your-username/Movie_Project`
   - **Branch**: `main` (or `master`)
   - **Main file path**: `recommender.py`
4. Click **Deploy!** 🚀

*Your app will be live 24/7 on a free URL like `https://cinematch-pro.streamlit.app`.*

---

## ✨ Features

- **🔥 Trending Highlights**: Weekly top trending movies fetched dynamically from TMDB API with a featured Hero banner.
- **🎯 Smart AI Recommendations**: Enhanced TF-IDF vectorizer (with unigrams & bigrams) + Cosine Similarity engine. Boosts director, cast, and genre matches for high accuracy.
- **🎬 Interactive Details & Embedded Trailers**: In-page details overlay featuring embedded YouTube video player (`st.video`), director, cast list, runtime, vote count, and full plot synopsis.
- **⭐ Persistent Watchlist**: Bookmark favorite movies to `user_data.json` with direct one-click recommendation triggers.
- **🔍 Advanced Filter Engine**: Narrow recommendations or watchlist by 19 genres, minimum TMDB rating (`⭐ 0.0 - 10.0`), and release year range (`1950 - 2026`).
- **🎨 Glassmorphism Dark UI**: Netflix/Apple TV+ inspired dark glassmorphism theme, genre pill badges, star rating badges, glowing card hover animations, and custom scrollbars.

---

## 📂 Project Structure

```
Movie_Project/
├── data/
│   ├── tmdb_5000_movies.csv      # Raw TMDB Movies Dataset
│   └── tmdb_5000_credits.csv     # Raw TMDB Credits Dataset
├── recommender.py                 # Main Streamlit Web Application
├── setup.py                       # Precomputation Script (TF-IDF Similarity Matrix)
├── user_data.json                 # Persistent Watchlist Data Store
├── requirements.txt               # Cleaned Python Dependencies
├── .env.example                   # Environment Configuration Template
├── index.html                     # GitHub Pages Landing Page
├── movies.pkl                     # Processed Movies DataFrame Cache
├── similarity.pkl                 # Precalculated Cosine Similarity Matrix Cache
└── README.md                      # Comprehensive Project Documentation
```

---

## ⚡ Local Setup Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Precompute Movie Similarity Matrix
Run the precomputation script to generate binary caches (`movies.pkl` and `similarity.pkl`):
```bash
python setup.py
```

### 4. Run the Web Application
Launch the Streamlit app server:
```bash
streamlit run recommender.py
```

Open your browser at `http://localhost:8501`.

---

## 🔑 Environment Configuration (Optional)

You can specify a custom TMDB API Key by setting the `TMDB_API_KEY` environment variable:

```bash
# Windows PowerShell
$env:TMDB_API_KEY="your_api_key_here"

# Linux / macOS
export TMDB_API_KEY="your_api_key_here"
```

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** — Web UI Framework
- **[Pandas](https://pandas.pydata.org/)** — Data Manipulation & Preprocessing
- **[Scikit-Learn](https://scikit-learn.org/)** — TF-IDF Vectorization & Cosine Similarity
- **[TMDB API](https://www.themoviedb.org/documentation/api)** — Live Movie Posters, Backdrops, Metadata & Video Trailers
