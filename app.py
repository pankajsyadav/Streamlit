import os
import re
import zipfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


# ─── Utilities ────────────────────────────────────────────────────────────────

def normalize_title(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def minmax_series(series):
    if series.nunique() <= 1:
        return pd.Series(0.0, index=series.index)
    from sklearn.preprocessing import MinMaxScaler
    values = MinMaxScaler().fit_transform(series.values.reshape(-1, 1)).flatten()
    return pd.Series(values, index=series.index)


def genre_overlap_score(movie_genres_str, selected_genres):
    if not selected_genres or not pd.notna(movie_genres_str):
        return 0.0
    return len(set(movie_genres_str.split("|")) & set(selected_genres)) / len(set(selected_genres))


# ─── Data & Model Loading ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data and building models...")
def build_models(data_dir="data", min_ratings=20):
    movies  = pd.read_csv(f"{data_dir}/movies.csv")
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    tags    = pd.read_csv(f"{data_dir}/tags.csv")

    movies["title_norm"] = movies["title"].apply(normalize_title)
    rating_counts = ratings.groupby("movieId")["rating"].count().rename("rating_count")
    movies = movies.merge(rating_counts, on="movieId", how="left")
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)

    id_to_title = dict(zip(movies["movieId"], movies["title"]))
    title_to_id = dict(zip(movies["title_norm"], movies["movieId"]))

    all_genres = sorted({
        g for gs in movies["genres"].fillna("") for g in gs.split("|")
        if g and g != "(no genres listed)"
    })

    eligible_ids = set(movies.loc[movies["rating_count"] >= min_ratings, "movieId"])
    user_item = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    eligible_cols = [c for c in user_item.columns if c in eligible_ids]
    user_mean = user_item[eligible_cols].mean(axis=1)
    movie_user = user_item[eligible_cols].sub(user_mean, axis=0).fillna(0).T
    cf_sim_df = pd.DataFrame(cosine_similarity(movie_user), index=movie_user.index, columns=movie_user.index)

    tags["tag"] = tags["tag"].fillna("").astype(str).str.lower().str.strip()
    tags_agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join([v for v in x if v])).rename("all_tags")
    content_df = movies[["movieId", "title", "genres", "title_norm"]].merge(tags_agg, on="movieId", how="left")
    content_df["all_tags"] = content_df["all_tags"].fillna("")
    content_df["content_text"] = (
        content_df["genres"].fillna("").str.replace("|", " ", regex=False).str.lower()
        + " " + content_df["all_tags"]
    ).str.strip()

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    content_matrix = tfidf.fit_transform(content_df["content_text"])
    content_sim_df = pd.DataFrame(
        cosine_similarity(content_matrix),
        index=content_df["movieId"],
        columns=content_df["movieId"]
    )

    return movies, id_to_title, title_to_id, all_genres, cf_sim_df, content_sim_df


# ─── Recommender ──────────────────────────────────────────────────────────────

def recommend(movies, title_to_id, id_to_title, cf_sim_df, content_sim_df,
              selected_genres, selected_movies, top_n=10,
              weight_cf=0.45, weight_content=0.35, weight_genre=0.20):

    seed_ids = [title_to_id[normalize_title(t)] for t in selected_movies
                if normalize_title(t) in title_to_id]

    if selected_genres:
        mask = movies["genres"].apply(
            lambda g: any(genre in str(g).split("|") for genre in selected_genres)
        )
        candidate_ids = sorted(set(movies.loc[mask, "movieId"]) - set(seed_ids))
    else:
        candidate_ids = sorted(set(movies["movieId"]) - set(seed_ids))

    if not candidate_ids:
        return pd.DataFrame()

    scores = pd.DataFrame(index=candidate_ids)

    cf_seeds = [m for m in seed_ids if m in cf_sim_df.index]
    scores["cf_score"] = minmax_series(
        cf_sim_df.loc[cf_seeds].mean(axis=0).reindex(candidate_ids).fillna(0.0)
        if cf_seeds else pd.Series(0.0, index=candidate_ids)
    )

    ct_seeds = [m for m in seed_ids if m in content_sim_df.index]
    scores["content_score"] = minmax_series(
        content_sim_df.loc[ct_seeds].mean(axis=0).reindex(candidate_ids).fillna(0.0)
        if ct_seeds else pd.Series(0.0, index=candidate_ids)
    )

    genre_lookup = movies.set_index("movieId")["genres"]
    scores["genre_score"] = pd.Series({
        mid: genre_overlap_score(genre_lookup.get(mid, ""), selected_genres)
        for mid in candidate_ids
    })

    if not seed_ids:
        weight_cf = 0.0
        total = (weight_content + weight_genre) or 1.0
        weight_content /= total
        weight_genre   /= total

    scores["final_score"] = (
        weight_cf * scores["cf_score"] +
        weight_content * scores["content_score"] +
        weight_genre * scores["genre_score"]
    )

    top = (
        scores.nlargest(top_n, "final_score")
        .reset_index().rename(columns={"index": "movieId"})
        .merge(movies[["movieId", "title", "genres", "rating_count"]], on="movieId", how="left")
    )
    return top[["title", "genres", "rating_count", "cf_score", "content_score", "genre_score", "final_score"]]


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="MovieLens Recommender", layout="wide")
st.title("🎬 MovieLens Movie Recommender")
st.caption("Select genres you enjoy and a few movies you like, then hit Recommend.")

movies, id_to_title, title_to_id, all_genres, cf_sim_df, content_sim_df = build_models()

col1, col2 = st.columns([1, 2])

with col1:
    selected_genres = st.multiselect(
        "1. Choose genres you enjoy", all_genres,
        default=["Animation", "Comedy"]
    )

with col2:
    selected_movies = st.multiselect(
        "2. Choose a few movies you like",
        sorted(movies["title"].tolist()),
        default=["Toy Story (1995)"]
    )

st.write("---")
st.subheader("Model Weights")
w_cf      = st.slider("Audience behavior weight", 0.0, 1.0, 0.45, 0.05)
w_content = st.slider("Content weight (genres & tags)",           0.0, 1.0, 0.35, 0.05)
w_genre   = st.slider("Genre preference weight",                  0.0, 1.0, 0.20, 0.05)

total_w = w_cf + w_content + w_genre
if total_w == 0:
    st.warning("At least one weight must be > 0.")
else:
    w_cf /= total_w; w_content /= total_w; w_genre /= total_w

if st.button("🎯 Recommend", type="primary"):
    if not selected_genres and not selected_movies:
        st.warning("Please select at least one genre or one movie.")
    else:
        recs = recommend(
            movies, title_to_id, id_to_title, cf_sim_df, content_sim_df,
            selected_genres, selected_movies,
            top_n=10, weight_cf=w_cf, weight_content=w_content, weight_genre=w_genre
        )
        if recs.empty:
            st.error("No recommendations found. Try different genres or movies.")
        else:
            st.success(f"Top {len(recs)} recommendations")
            st.dataframe(recs, use_container_width=True, hide_index=True)

st.write("---")
st.caption(
    "Data: MovieLens small dataset (Harper & Konstan, 2015, https://doi.org/10.1145/2827872). "
    "Method inspired by Memphis Meng (2020), Towards Data Science."
)
