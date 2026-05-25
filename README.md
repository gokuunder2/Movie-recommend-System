# 🎬 Movie Recommendation System

A content-based movie recommendation system built using the **TMDB 5000 Movies dataset**. Given a movie title, the system recommends 5 similar movies by analysing genres, keywords, cast, crew, and plot overview using NLP and cosine similarity.

---

## How it works

The system builds a "tag" for each movie by combining its overview, genres, keywords, top 3 cast members, and director into a single text blob. It then vectorises these tags using Bag of Words and measures similarity between movies using cosine similarity — the closer the angle between two vectors, the more similar the movies.

```
Movie Metadata → Tag Creation → Stemming → Count Vectorization → Cosine Similarity → Top 5 Recommendations
```

---

## Dataset

| File | Source |
|------|--------|
| `tmdb_5000_movies.csv` | [TMDB 5000 Movie Dataset – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |
| `tmdb_5000_credits.csv` | Same Kaggle dataset (credits file) |

Both files are merged on the `title` column. The relevant columns kept are: `id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`.

---

## Tech stack

- **Python 3**
- **pandas** — data loading and manipulation
- **NumPy** — array operations
- **scikit-learn** — `CountVectorizer` for Bag of Words, `cosine_similarity` for distance calculation
- **NLTK** — `PorterStemmer` for stemming tags
- **ast** — parsing JSON-like strings in the dataset columns
- **Google Colab** — notebook environment with Google Drive mount

---

## Project structure

```
movie_recommend_system.ipynb   ← main notebook (run in Google Colab)
tmdb_5000_movies.csv.zip       ← upload to Google Drive before running
tmdb_5000_credits.csv.zip      ← upload to Google Drive before running
```

---

## Setup & usage

### 1. Clone the repo
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
```

### 2. Upload datasets to Google Drive
Upload both `tmdb_5000_movies.csv.zip` and `tmdb_5000_credits.csv.zip` to your Google Drive root (`MyDrive/`).

### 3. Open in Google Colab
Open `movie_recommend_system.ipynb` in Google Colab and mount your Drive when prompted.

### 4. Run all cells
Run the notebook top to bottom. Once complete, call the `recommend()` function:

```python
recommend('Gandhi')
# Output:
# The Legend of Bhagat Singh
# Mangal Pandey: The Rising
# ...
```

---

## Key steps in the notebook

1. **Data loading** — reads both CSVs from Google Drive and merges them on `title`
2. **Feature selection** — keeps 7 columns: `id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`
3. **Data cleaning** — drops nulls and duplicates
4. **Feature extraction** — parses JSON strings using `ast.literal_eval` to extract genre names, keyword names, top 3 cast names, and the director's name from crew
5. **Space removal** — collapses multi-word names (e.g. `Sam Mendes` → `SamMendes`) so the vectoriser treats them as one token
6. **Tag creation** — concatenates overview + genres + keywords + cast + crew into a single `tags` column
7. **Stemming** — applies `PorterStemmer` to reduce words to their root form (`loves` → `love`)
8. **Vectorisation** — `CountVectorizer` with `max_features=5000` and English stop words removed builds a word frequency matrix
9. **Similarity matrix** — `cosine_similarity` computes pairwise similarity scores across all ~4800 movies
10. **Recommendation** — `recommend(movie)` looks up the movie's index, sorts all movies by similarity score, and returns the top 5

---

## Example output

```python
recommend('The Dark Knight')
# Batman Begins
# The Dark Knight Rises
# Batman
# Batman Forever
# Batman & Robin
```

---

## Limitations

- Content-based only — no collaborative filtering, so it doesn't factor in user ratings or viewing history
- Recommendations are driven heavily by cast/director overlap; two unrelated films with the same director may be suggested
- The similarity matrix is computed fully in memory — not scalable to very large datasets without approximate nearest neighbour methods (e.g. FAISS)

---

## Future improvements

- Add TF-IDF vectorisation instead of raw Bag of Words for better weighting
- Build a Streamlit or Flask web app UI
- Incorporate collaborative filtering using user ratings
- Persist the similarity matrix with `pickle` to avoid recomputing on every run

---

## License

This project is for educational purposes. Dataset credit: [The Movie Database (TMDB)](https://www.themoviedb.org/).
