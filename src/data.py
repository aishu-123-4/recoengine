"""Load MovieLens 100K ratings and movie metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings() -> pd.DataFrame:
    path = DATA / "u.data"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.download_data")
    return pd.read_csv(
        path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )


def load_movies() -> pd.DataFrame:
    path = DATA / "u.item"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.download_data")
    cols = ["item_id", "title", "release_date", "video_date", "imdb_url"] + GENRES
    df = pd.read_csv(path, sep="|", names=cols, encoding="latin-1")
    return df


def title_lookup() -> dict[int, str]:
    movies = load_movies()
    return dict(zip(movies["item_id"], movies["title"]))
