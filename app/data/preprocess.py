from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.config import AppConfig
from app.data.loader import load_raw_data, validate_raw_data
from app.logger import get_logger
from app.utils import ensure_directories

logger = get_logger(__name__)


def _build_user_profiles(interactions_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    merged = interactions_df.merge(
        products_df[["product_id", "category"]],
        on="product_id",
        how="left",
        suffixes=("", "_product"),
    )

    if "category_product" in merged.columns:
        merged["category"] = merged["category_product"]
        merged = merged.drop(columns=["category_product"])

    category_pref = (
        merged.groupby(["user_id", "category"])["event_weight"]
        .sum()
        .reset_index(name="category_weight")
    )

    category_total = category_pref.groupby("user_id")["category_weight"].transform("sum")
    category_pref["category_preference_score"] = category_pref["category_weight"] / category_total

    pivot = category_pref.pivot(
        index="user_id",
        columns="category",
        values="category_preference_score",
    ).fillna(0.0)

    pivot.columns = [f"pref_{col}" for col in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


def _build_product_popularity(interactions_df: pd.DataFrame) -> pd.DataFrame:
    popularity_df = (
        interactions_df.groupby("product_id")
        .agg(
            interaction_count=("product_id", "count"),
            weighted_popularity=("event_weight", "sum"),
        )
        .reset_index()
    )

    scaler = MinMaxScaler()
    popularity_df["normalized_popularity"] = scaler.fit_transform(
        popularity_df[["weighted_popularity"]]
    )

    return popularity_df


def _build_user_product_scores(interactions_df: pd.DataFrame) -> pd.DataFrame:
    user_product_scores = (
        interactions_df.groupby(["user_id", "product_id"])
        .agg(
            interaction_count=("product_id", "count"),
            user_product_score=("event_weight", "sum"),
            last_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
    return user_product_scores


def preprocess_and_save(config: AppConfig) -> None:
    ensure_directories(
        [
            config.paths.raw_data_dir,
            config.paths.processed_data_dir,
            config.paths.artifacts_dir,
            config.paths.models_dir,
        ]
    )

    users_df, products_df, interactions_df = load_raw_data(config)
    validate_raw_data(users_df, products_df, interactions_df)

    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    interactions_df = interactions_df.sort_values("timestamp").reset_index(drop=True)

    user_profiles_df = _build_user_profiles(interactions_df, products_df)
    product_popularity_df = _build_product_popularity(interactions_df)
    user_product_scores_df = _build_user_product_scores(interactions_df)

    users_processed = users_df.merge(user_profiles_df, on="user_id", how="left").fillna(0.0)
    products_processed = products_df.merge(product_popularity_df, on="product_id", how="left").fillna(0.0)
    interactions_processed = interactions_df.copy()

    users_processed.to_csv(config.paths.processed_data_dir / "users_processed.csv", index=False)
    products_processed.to_csv(config.paths.processed_data_dir / "products_processed.csv", index=False)
    interactions_processed.to_csv(config.paths.processed_data_dir / "interactions_processed.csv", index=False)
    user_product_scores_df.to_csv(
        config.paths.processed_data_dir / "user_product_scores.csv",
        index=False,
    )

    metadata = {
        "n_users": int(len(users_processed)),
        "n_products": int(len(products_processed)),
        "n_interactions": int(len(interactions_processed)),
        "user_profile_columns": [col for col in users_processed.columns if col.startswith("pref_")],
    }

    with open(config.paths.processed_data_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(metadata, config.paths.models_dir / "preprocess_metadata.joblib")

    logger.info("Saved processed users to %s", config.paths.processed_data_dir / "users_processed.csv")
    logger.info("Saved processed products to %s", config.paths.processed_data_dir / "products_processed.csv")
    logger.info(
        "Saved processed interactions to %s",
        config.paths.processed_data_dir / "interactions_processed.csv",
    )
    logger.info("Saved user-product scores to %s", config.paths.processed_data_dir / "user_product_scores.csv")
    logger.info("Preprocessing completed successfully")
