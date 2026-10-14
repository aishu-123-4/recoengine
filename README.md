# RecoEngine — Movie Recommender System  🎬

A recommender system built **three ways** on the classic **MovieLens 100K**
dataset, compared head-to-head with proper ranking metrics:

- **Item-item collaborative filtering** (with significance weighting)
- **SVD matrix factorization** (latent factors)
- **Content-based** (genre-profile matching)
- **Hybrid** (behavioral + content blend)

## Dataset

MovieLens 100K (GroupLens): **100,000 ratings** (1–5) from **943 users** on
**1,682 movies**, plus movie genres. Fetched by `src/download_data.py`.

## Results

Per-user train/test split (80% / 20%), top-10 recommendations, a held-out movie
counts as relevant if the user rated it ≥ 4. Evaluated on 300 users:

| Model | precision@10 | recall@10 | NDCG@10 |
|-------|:------------:|:---------:|:-------:|
| **Item-item CF** | **0.219** | **0.233** | **0.309** |
| SVD (40 factors) | 0.159 | 0.154 | 0.215 |
| Hybrid (0.8·SVD + 0.2·content) | 0.155 | 0.147 | 0.211 |
| Content-based | 0.015 | 0.017 | 0.021 |

### Reading the results

- **Item-item CF wins** on this dataset — MovieLens users have dense histories,
  so "people who liked what you liked" is a strong signal. precision@10 ≈ 0.22
  is in line with published MovieLens baselines.
- **Content-based is weak alone** (0.015): genres are too coarse to rank well by
  themselves. Its real value is **cold-start** — it can recommend a brand-new
  movie with zero ratings purely from its genres, which CF and SVD cannot.
- **The hybrid** keeps most of SVD's accuracy while folding in content, trading a
  little precision for cold-start robustness. The blend weight `alpha` is tunable.

### Why the item-CF tuning matters

Naïve cosine item-CF scored **~0** here: obscure movies with a single co-rating
get similarity 1.0 and flood the top-10. Two fixes brought it to 0.22:

1. **Significance weighting** — shrink each pair's similarity by
   `co_ratings / (co_ratings + 25)`, so thinly-supported pairs count for little.
2. **Rank by the raw similarity-weighted sum**, not a normalized predicted
   rating (normalizing cancels the support signal and re-promotes cold items).

This debugging story is exactly the kind of practical ML judgment interviews probe.

## Quickstart

```bash
cd 02-recoengine
pip install -r requirements.txt

python -m src.download_data          # fetch MovieLens 100K into data/
python -m src benchmark              # compare all 4 models
python -m src recommend --user 196 --model itemcf   # top-10 for a user
python -m src similar --title "Star Wars"           # content-similar movies
```

Example:

```
Top 5 for user 196 (itemcf):
  1. Back to the Future (1985)
  2. Star Wars (1977)
  3. When Harry Met Sally... (1989)
  4. Fargo (1996)
  5. Raiders of the Lost Ark (1981)

Movies similar to 'Star Wars (1977)':
  - Return of the Jedi (1983)
  - Empire Strikes Back, The (1980)
  - Starship Troopers (1997)
  - Star Trek: First Contact (1996)
```

## Project layout

```
02-recoengine/
├── README.md
├── requirements.txt
├── data/                 # MovieLens files (git-ignored)
└── src/
    ├── download_data.py
    ├── data.py           # load ratings + movie/genre metadata
    ├── recommenders.py   # ItemCF / SVD / ContentBased / Hybrid
    ├── evaluate.py       # split + precision@k / recall@k / NDCG@k
    └── __main__.py       # benchmark / recommend / similar CLI
tests/
└── test_recoengine.py    # 9 tests on a synthetic rating set (no download)
```

## Tests

```bash
python -m pytest -q      # 9 tests
```

## Possible extensions

- Implement **BPR** or **ALS** implicit-feedback models.
- Add a bias baseline (global + user + item biases) and funkSVD with SGD.
- Time-aware split (train on older ratings, test on newer) for a realistic setup.
- A small Streamlit UI for interactive recommendations.

## Credit

MovieLens 100K, GroupLens Research, University of Minnesota. Used for
research/education.
