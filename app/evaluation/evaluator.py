from __future__ import annotations

import pandas as pd

from app.config import AppConfig
from app.evaluation.metrics import hit_rate_at_k, precision_at_k, recall_at_k
from app.logger import get_logger
from app.recommender.service import RecommendationService, load_processed_data

logger = get_logger(__name__)


def _temporal_train_test_split(
    interactions_df: pd.DataFrame,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for _, user_group in interactions_df.groupby("user_id"):
        user_group = user_group.sort_values("timestamp")
        n = len(user_group)

        if n < 2:
            train_parts.append(user_group)
            continue

        split_idx = max(1, int(n * (1 - test_ratio)))
        split_idx = min(split_idx, n - 1)

        train_parts.append(user_group.iloc[:split_idx])
        test_parts.append(user_group.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=interactions_df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=interactions_df.columns)
    return train_df, test_df


def evaluate_recommender(config: AppConfig, k: int) -> dict:
    users_df, products_df, interactions_df = load_processed_data(config)
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    interactions_df = interactions_df.sort_values("timestamp").reset_index(drop=True)

    train_df, test_df = _temporal_train_test_split(
        interactions_df=interactions_df,
        test_ratio=config.evaluation.test_ratio,
    )

    eligible_users = []
    train_counts = train_df.groupby("user_id").size().to_dict()
    test_counts = test_df.groupby("user_id").size().to_dict()

    for user_id in users_df["user_id"].tolist():
        if (
            train_counts.get(user_id, 0) >= config.evaluation.min_user_train_interactions
            and test_counts.get(user_id, 0) >= config.evaluation.min_user_test_interactions
        ):
            eligible_users.append(user_id)

    if not eligible_users:
        raise ValueError("No eligible users found for evaluation")

    service = RecommendationService(
        users_df=users_df,
        products_df=products_df,
        interactions_df=train_df,
        config=config,
    )

    precision_scores = []
    recall_scores = []
    hit_rate_scores = []

    for user_id in eligible_users:
        actual_items = (
            test_df[test_df["user_id"] == user_id]["product_id"].dropna().astype(str).unique().tolist()
        )

        recommendations = service.recommend_for_user(user_id=user_id, top_k=k)
        predicted_items = recommendations["product_id"].astype(str).tolist()

        precision_scores.append(precision_at_k(actual_items, predicted_items, k))
        recall_scores.append(recall_at_k(actual_items, predicted_items, k))
        hit_rate_scores.append(hit_rate_at_k(actual_items, predicted_items, k))

    metrics = {
        "evaluated_users": len(eligible_users),
        "precision_at_k": round(sum(precision_scores) / len(precision_scores), 4),
        "recall_at_k": round(sum(recall_scores) / len(recall_scores), 4),
        "hit_rate_at_k": round(sum(hit_rate_scores) / len(hit_rate_scores), 4),
        "k": k,
    }

    logger.info("Evaluation metrics: %s", metrics)
    return metrics
