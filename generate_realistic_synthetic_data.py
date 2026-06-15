"""
Generate realistic synthetic browsing logs for the MCMC portfolio project.

The first synthetic dataset was intentionally broad and balanced. This generator
creates more human-like browsing behavior:

- users have sparse preferences, usually 1-3 dominant categories;
- sessions are intent-driven and stay within a small category cluster;
- category popularity is skewed rather than uniform;
- occasional exploration and switching still exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TOP10 = np.array([5, 9, 19, 11, 20, 23, 6, 21, 25, 22])
CATEGORY_NAMES = {
    5: "Toys",
    9: "Home",
    19: "Vehicles",
    11: "Electronics",
    20: "Travel",
    23: "Appliances",
    6: "Books",
    21: "Mobile",
    25: "Lifestyle",
    22: "Gaming",
}

# Related-category clusters, indexed by original category ID.
RELATED = {
    5: [22, 6, 9],
    9: [23, 5, 25],
    19: [20, 11, 21],
    11: [21, 23, 22],
    20: [19, 9, 25],
    23: [11, 9, 21],
    6: [5, 22, 25],
    21: [11, 23, 22],
    25: [9, 20, 6],
    22: [5, 11, 21],
}


def category_popularity():
    # Skewed market-level popularity. Sums to 1 after normalization.
    weights = np.array([0.09, 0.18, 0.06, 0.15, 0.08, 0.12, 0.09, 0.11, 0.05, 0.07])
    return weights / weights.sum()


def build_user_profile(rng: np.random.Generator):
    primary = int(rng.choice(TOP10, p=category_popularity()))

    related_pool = RELATED[primary]
    secondary_count = int(rng.choice([1, 2], p=[0.7, 0.3]))
    secondary = list(rng.choice(related_pool, size=secondary_count, replace=False))

    # Sparse preference mass: most users focus on one category and a few related ones.
    weights = {int(cat): 0.0 for cat in TOP10}
    weights[primary] = float(rng.uniform(0.58, 0.78))
    remaining = 1.0 - weights[primary]

    secondary_mass = float(rng.uniform(0.16, min(0.32, remaining)))
    split = rng.dirichlet(np.ones(len(secondary)))
    for cat, share in zip(secondary, split):
        weights[int(cat)] += secondary_mass * float(share)

    exploration_cats = [int(c) for c in TOP10 if int(c) not in [primary] + secondary]
    exploration_mass = max(1.0 - sum(weights.values()), 0.0)
    explore_split = rng.dirichlet(np.ones(len(exploration_cats)) * 0.6)
    for cat, share in zip(exploration_cats, explore_split):
        weights[cat] += exploration_mass * float(share)

    prob = np.array([weights[int(c)] for c in TOP10])
    prob = prob / prob.sum()
    return primary, secondary, prob


def sample_session_sequence(rng, primary, secondary, user_prob):
    length = int(np.clip(rng.lognormal(mean=1.75, sigma=0.45), 3, 18))

    intent_type = rng.choice(["primary", "secondary", "explore"], p=[0.74, 0.21, 0.05])
    if intent_type == "primary":
        intent = primary
    elif intent_type == "secondary" and secondary:
        intent = int(rng.choice(secondary))
    else:
        intent = int(rng.choice(TOP10, p=user_prob))

    seq = []
    current = intent
    for t in range(length):
        if t == 0:
            current = intent
        else:
            move = rng.choice(["stay", "related", "profile", "random"], p=[0.68, 0.19, 0.10, 0.03])
            if move == "stay":
                current = current
            elif move == "related":
                current = int(rng.choice(RELATED[int(current)]))
            elif move == "profile":
                current = int(rng.choice(TOP10, p=user_prob))
            else:
                current = int(rng.choice(TOP10))
        seq.append(int(current))
    return seq


def generate(root: Path, n_users: int = 5200, seed: int = 20260614):
    rng = np.random.default_rng(seed)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    session_counter = 1
    base_train = pd.Timestamp("2022-01-01")
    base_test = pd.Timestamp("2022-02-01")

    for u_idx in range(n_users):
        cid = 100000 + u_idx
        primary, secondary, user_prob = build_user_profile(rng)

        # Keep every user in both periods, but let activity level vary.
        activity = rng.choice(["low", "medium", "high"], p=[0.35, 0.50, 0.15])
        if activity == "low":
            n_train_sessions = int(rng.integers(1, 3))
            n_test_sessions = int(rng.integers(1, 2))
        elif activity == "medium":
            n_train_sessions = int(rng.integers(2, 5))
            n_test_sessions = int(rng.integers(1, 3))
        else:
            n_train_sessions = int(rng.integers(4, 8))
            n_test_sessions = int(rng.integers(2, 5))

        for split, n_sessions, base_date, max_day in [
            ("train", n_train_sessions, base_train, 28),
            ("test", n_test_sessions, base_test, 85),
        ]:
            for _ in range(n_sessions):
                session_id = f"{split}_{session_counter:07d}"
                session_counter += 1
                day_offset = int(rng.integers(0, max_day))
                minute_offset = int(rng.integers(8 * 60, 24 * 60))
                start_time = base_date + pd.Timedelta(days=day_offset, minutes=minute_offset)
                seq = sample_session_sequence(rng, primary, secondary, user_prob)

                for order, category in enumerate(seq):
                    rows.append(
                        {
                            "ID": cid,
                            "session_id": session_id,
                            "order": order,
                            "time": start_time + pd.Timedelta(seconds=35 * order + int(rng.integers(0, 25))),
                            "category": int(category),
                            "event": 1,
                            "category_name": CATEGORY_NAMES[int(category)],
                            "primary_interest": CATEGORY_NAMES[int(primary)],
                            "is_fake_data": True,
                            "synthetic_version": "realistic_sparse_v1",
                        }
                    )

    df = pd.DataFrame(rows).sort_values(["ID", "session_id", "order"]).reset_index(drop=True)
    parquet_path = data_dir / "itemPV_202201to202204.parquet"
    pkl_path = data_dir / "itemPV_202201to202204.pkl"
    df.to_parquet(parquet_path, index=False)
    df.to_pickle(pkl_path)
    df.head(1000).to_csv(data_dir / "sample_preview.csv", index=False, encoding="utf-8-sig")

    summary = {
        "folder": str(root).replace("\\", "/"),
        "synthetic_version": "realistic_sparse_v1",
        "primary_data_file": "data/itemPV_202201to202204.parquet",
        "primary_data_size_bytes": parquet_path.stat().st_size,
        "rows": int(len(df)),
        "users": int(df["ID"].nunique()),
        "sessions": int(df["session_id"].nunique()),
        "train_rows": int((df["time"] < pd.Timestamp("2022-02-01")).sum()),
        "test_rows": int((df["time"] >= pd.Timestamp("2022-02-01")).sum()),
        "events_per_session_mean": float(df.groupby("session_id").size().mean()),
        "median_unique_categories_per_user": float(df.groupby("ID")["category"].nunique().median()),
        "category_counts": {str(k): int(v) for k, v in df["category"].value_counts().sort_index().items()},
        "primary_interest_counts": {str(k): int(v) for k, v in df.groupby("ID")["primary_interest"].first().value_counts().sort_index().items()},
    }
    (root / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    root = Path(__file__).resolve().parent
    summary = generate(root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
