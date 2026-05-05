
from __future__ import annotations

import pandas as pd

from app.config import AppConfig
from app.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users_path = config.paths.raw_data_dir / "users.csv"
    products_path = config.paths.raw_data_dir / "products.csv"
    interactions_path = config.paths.raw_data_dir / "interactions.csv"

    users_df = pd.read_csv(users_path)
    products_df = pd.read_csv(products_path)
    interactions_df = pd.read_csv(interactions_path)

    logger.info(
        "Loaded raw data | users=%d | products=%d | interactions=%d",
        len(users_df),
        len(products_df),
        len(interactions_df),
    )
    return users_df, products_df, interactions_df


def validate_raw_data(
    users_df: pd.DataFrame,
    products_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
) -> None:
    required_user_cols = {
        "user_id",
        "age_group",
        "gender",
        "price_sensitivity",
        "activity_level",
        "preferred_categories",
    }
    required_product_cols = {"product_id", "name", "category", "brand", "price", "tags"}
    required_interaction_cols = {
        "user_id",
        "product_id",
        "event_type",
        "event_weight",
        "category",
        "timestamp",
    }

    if not required_user_cols.issubset(set(users_df.columns)):
        missing = required_user_cols.difference(set(users_df.columns))
        raise ValueError(f"Missing user columns: {missing}")

    if not required_product_cols.issubset(set(products_df.columns)):
        missing = required_product_cols.difference(set(products_df.columns))
        raise ValueError(f"Missing product columns: {missing}")

    if not required_interaction_cols.issubset(set(interactions_df.columns)):
        missing = required_interaction_cols.difference(set(interactions_df.columns))
        raise ValueError(f"Missing interaction columns: {missing}")

    if users_df["user_id"].duplicated().any():
        raise ValueError("Duplicate user_id detected")

    if products_df["product_id"].duplicated().any():
        raise ValueError("Duplicate product_id detected")

    user_ids = set(users_df["user_id"])
    product_ids = set(products_df["product_id"])

    if not set(interactions_df["user_id"]).issubset(user_ids):
        raise ValueError("Interactions contain unknown user_id values")

    if not set(interactions_df["product_id"]).issubset(product_ids):
        raise ValueError("Interactions contain unknown product_id values")

    if users_df.isna().any().any():
        raise ValueError("Users data contains missing values")

    if products_df.isna().any().any():
        raise ValueError("Products data contains missing values")

    if interactions_df.isna().any().any():
        raise ValueError("Interactions data contains missing values")

    logger.info("Raw data validation passed")
