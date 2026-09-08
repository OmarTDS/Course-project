import pickle

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Model definition — must match the architecture trained in the notebook.
# ---------------------------------------------------------------------------
class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, user_idx, movie_idx):
        user_vec = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)

        x = torch.cat([user_vec, movie_vec], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return 1 + 4 * torch.sigmoid(x.squeeze())


# ---------------------------------------------------------------------------
# Cached loaders — Streamlit reruns the whole script on every interaction,
# so caching keeps us from reloading the model and data on every click.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("app_data.pkl", "rb") as f:
        data = pickle.load(f)

    model = RecommenderNet(data["n_users"], data["n_movies"], embedding_dim=50)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    return model, data


def create_new_user_embedding(model, quiz_movie_indices, quiz_ratings, n_epochs=200, lr=0.01):
    for param in model.parameters():
        param.requires_grad = False

    new_user_vec = torch.randn(1, model.user_embedding.embedding_dim, requires_grad=True)
    optimizer = optim.Adam([new_user_vec], lr=lr)
    criterion = nn.MSELoss()

    quiz_movie_tensor = torch.tensor(quiz_movie_indices, dtype=torch.long)
    quiz_rating_tensor = torch.tensor(quiz_ratings, dtype=torch.float32)

    for _ in range(n_epochs):
        optimizer.zero_grad()

        movie_vecs = model.movie_embedding(quiz_movie_tensor)
        user_vecs = new_user_vec.repeat(len(quiz_movie_indices), 1)

        x = torch.cat([user_vecs, movie_vecs], dim=1)
        x = model.relu(model.fc1(x))
        x = model.relu(model.fc2(x))
        x = model.fc3(x)
        pred = 1 + 4 * torch.sigmoid(x.squeeze())

        loss = criterion(pred, quiz_rating_tensor)
        loss.backward()
        optimizer.step()

    for param in model.parameters():
        param.requires_grad = True

    return new_user_vec.detach()


def get_recommendations_new_user(
    user_vec, model, n_movies, movies_df, movie_to_idx,
    quiz_movie_indices, ratings_df, top_n=10, min_ratings=20,
):
    model.eval()

    idx_to_movie = {v: k for k, v in movie_to_idx.items()}
    already_rated = set(quiz_movie_indices)
    movie_counts = ratings_df.groupby("movie_idx").size()

    all_movie_indices = torch.arange(n_movies)
    movie_vecs = model.movie_embedding(all_movie_indices)
    user_vecs = user_vec.repeat(n_movies, 1)

    with torch.no_grad():
        x = torch.cat([user_vecs, movie_vecs], dim=1)
        x = model.relu(model.fc1(x))
        x = model.relu(model.fc2(x))
        x = model.fc3(x)
        predicted_ratings = 1 + 4 * torch.sigmoid(x.squeeze())

    predicted_ratings = predicted_ratings.numpy()

    results = []
    for movie_idx, pred in enumerate(predicted_ratings):
        if movie_idx in already_rated:
            continue
        if movie_counts.get(movie_idx, 0) < min_ratings:
            continue
        real_movie_id = idx_to_movie[movie_idx]
        title = movies_df[movies_df["movieId"] == real_movie_id]["title"].values[0]
        results.append((title, pred))

    results.sort(key=lambda pair: pair[1], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Find your next movie")
st.write(
    "Rate the movies you've seen below (skip the ones you haven't). "
    "We'll build a taste profile from your answers and recommend movies "
    "you haven't rated."
)

model, data = load_model()
quiz_movies = data["quiz_movies"]
quiz_movie_indices = data["quiz_movie_indices"]

st.divider()

seen = {}
ratings_input = {}

for title in quiz_movies:
    col1, col2 = st.columns([3, 2])
    with col1:
        seen[title] = st.checkbox(f"I've seen {title}", key=f"seen_{title}")
    with col2:
        if seen[title]:
            ratings_input[title] = st.slider(
                "Your rating", 1.0, 5.0, 3.0, 0.5, key=f"rating_{title}"
            )

st.divider()

if st.button("Get my recommendations", type="primary"):
    rated_titles = [t for t in quiz_movies if seen.get(t)]

    if len(rated_titles) < 3:
        st.warning("Rate at least 3 movies so we have enough signal to work with.")
    else:
        rated_indices = [
            quiz_movie_indices[quiz_movies.index(t)] for t in rated_titles
        ]
        rated_scores = [ratings_input[t] for t in rated_titles]

        with st.spinner("Building your taste profile..."):
            new_user_vec = create_new_user_embedding(
                model, rated_indices, rated_scores
            )
            recommendations = get_recommendations_new_user(
                new_user_vec,
                model,
                data["n_movies"],
                data["movies_df"],
                data["movie_to_idx"],
                rated_indices,
                data["ratings_df"],
                top_n=10,
            )

        st.subheader("Your recommendations")
        for title, score in recommendations:
            st.write(f"**{title}** — predicted rating: {score:.2f}")
