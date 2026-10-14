"""RecoEngine CLI.

    python -m src benchmark                 # compare all 4 models (precision@k/NDCG)
    python -m src recommend --user 196      # top-10 for a user (hybrid)
    python -m src similar --title "Star Wars"   # content-similar movies
"""

from __future__ import annotations

import argparse
import sys

from .data import load_movies, load_ratings, title_lookup
from .evaluate import build_model, evaluate, train_test_split_per_user
from .recommenders import ContentBased, Dataset


def cmd_benchmark(args: argparse.Namespace) -> int:
    ratings, movies = load_ratings(), load_movies()
    train, test = train_test_split_per_user(ratings, seed=args.seed)
    print(f"Train {len(train):,} / Test {len(test):,} ratings | k={args.k}\n")
    header = f"{'model':10} {'precision@k':>12} {'recall@k':>10} {'ndcg@k':>8} {'users':>7}"
    print(header); print("-" * len(header))
    for name in ("itemcf", "svd", "content", "hybrid"):
        m = evaluate(name, train, test, movies, k=args.k, n_users=args.users, seed=args.seed)
        print(f"{name:10} {m['precision@k']:>12} {m['recall@k']:>10} "
              f"{m['ndcg@k']:>8} {m['n_users_evaluated']:>7}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    ratings, movies = load_ratings(), load_movies()
    titles = title_lookup()
    model, data = build_model(args.model, ratings, movies)
    if args.user not in data.u_index:
        print(f"User {args.user} not found.", file=sys.stderr)
        return 1
    recs = model.recommend(args.user, args.k)
    print(f"Top {args.k} for user {args.user} ({args.model}):")
    for i, item in enumerate(recs, 1):
        print(f"  {i:2}. {titles.get(item, '?')}")
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    ratings, movies = load_ratings(), load_movies()
    titles = title_lookup()
    data = Dataset(ratings)
    cb = ContentBased(data, movies).fit()
    matches = movies[movies["title"].str.contains(args.title, case=False, na=False)]
    if matches.empty:
        print(f"No movie matching '{args.title}'.", file=sys.stderr)
        return 1
    item_id = int(matches.iloc[0]["item_id"])
    if item_id not in data.i_index:
        print("Movie has no ratings in the dataset.", file=sys.stderr)
        return 1
    vec = cb.item_genres[data.i_index[item_id]]
    scores = cb.item_genres @ vec
    order = scores.argsort()[::-1]
    print(f"Movies similar to '{titles[item_id]}':")
    shown = 0
    for j in order:
        mid = int(data.item_ids[j])
        if mid == item_id:
            continue
        print(f"  - {titles.get(mid, '?')}")
        shown += 1
        if shown >= args.k:
            break
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="recoengine")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark"); b.set_defaults(func=cmd_benchmark)
    b.add_argument("--k", type=int, default=10)
    b.add_argument("--users", type=int, default=200)
    b.add_argument("--seed", type=int, default=42)

    r = sub.add_parser("recommend"); r.set_defaults(func=cmd_recommend)
    r.add_argument("--user", type=int, required=True)
    r.add_argument("--model", default="hybrid",
                   choices=["itemcf", "svd", "content", "hybrid"])
    r.add_argument("--k", type=int, default=10)

    s = sub.add_parser("similar"); s.set_defaults(func=cmd_similar)
    s.add_argument("--title", required=True)
    s.add_argument("--k", type=int, default=10)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
