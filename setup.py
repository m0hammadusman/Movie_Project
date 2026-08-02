import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("⏳ Loading Data...")
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

# Merge and Clean
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj): return [i['name'] for i in ast.literal_eval(obj)]
def convert3(obj): return [i['name'] for i in ast.literal_eval(obj)][:3]
def fetch_director(obj):
    return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']

print("⚙️ Processing & Weighting Tags...")
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

for col in ['genres', 'keywords', 'cast', 'crew']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

# Feature Weighting: Boost Director (3x), Genres (2x), Cast (2x)
weighted_genres = movies['genres'].apply(lambda x: x * 2)
weighted_director = movies['crew'].apply(lambda x: x * 3)
weighted_cast = movies['cast'].apply(lambda x: x * 2)

movies['tags'] = movies['overview'] + weighted_genres + movies['keywords'] + weighted_cast + weighted_director
new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

print("🧮 Calculating Enhanced TF-IDF Similarity Matrix...")
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
vectors = tfidf.fit_transform(new_df['tags'])
similarity = cosine_similarity(vectors)

print("💾 Saving Files...")
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ DONE! TF-IDF engine ready with enhanced accuracy.")