from __future__ import annotations

import joblib
import pandas as pd

from app.config import AppConfig
from app.logger import get_logger
from app.recommender.content_based import ContentBasedRecommender
from app.recommender.hybrid import HybridRecommender
from app.recommender.popularity import PopularityRecommender
from app.utils import ensure_directories

logger = get_logger(__name__)


class RecommendationService:
    def __init__(
        self,
        users_df: pd.DataFrame,
        products_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        config: AppConfig,
    ) -> None:
        self.users_df = users_df.copy()
        self.products_df = products_df.copy()
        self.interactions_df = interactions_df.copy()
        self.config = config

        self.popularity_recommender = PopularityRecommender(self.products_df)
        self.content_recommender = ContentBasedRecommender(
            users_df=self.users_df,
            products_df=self.products_df,
            interactions_df=self.interactions_df,
        )
        self.hybrid_recommender = HybridRecommender(config.hybrid_weights)

        self.user_seen_products = (
            self.interactions_df.groupby("user_id")["product_id"].apply(set).to_dict()
        )

        self.category_columns = [col for col in self.users_df.columns if col.startswith("pref_")]

    def _category_scores_for_user(self, user_id: str) -> pd.DataFrame:
        user_row = self.users_df[self.users_df["user_id"] == user_id]
        if user_row.empty or not self.category_columns:
            return pd.DataFrame(
                {
                    "product_id": self.products_df["product_id"],
                    "category_score": 0.0,
                }
            )

        user_row = user_row.iloc[0]
        category_preferences = {}
        for col in self.category_columns:
            category_name = col.replace("pref_", "")
            category_preferences[category_name] = float(user_row[col])

        scores = []
        for _, product in self.products_df.iterrows():
            category_score = category_preferences.get(product["category"], 0.0)
            scores.append({"product_id": product["product_id"], "category_score": category_score})

        return pd.DataFrame(scores)

    def _cold_start_recommendations(self, top_k: int) -> pd.DataFrame:
        popularity_scores = self.popularity_recommender.score()
        result = popularity_scores.merge(
            self.products_df[["product_id", "name", "category", "brand", "price"]],
            on="product_id",
            how="left",
        )
        result = result.sort_values("popularity_score", ascending=False).head(top_k).reset_index(drop=True)
        result["recommendation_reason"] = "cold_start_popularity"
        result = result.rename(columns={"popularity_score": "score"})
        return result[
            ["product_id", "name", "category", "brand", "price", "score", "recommendation_reason"]
        ]

    def recommend_for_user(self, user_id: str, top_k: int = 5) -> pd.DataFrame:
        known_user_ids = set(self.users_df["user_id"])
        if user_id not in known_user_ids:
            logger.warning("Unknown user_id=%s. Falling back to popularity recommender", user_id)
            return self._cold_start_recommendations(top_k)

        user_history = self.interactions_df[self.interactions_df["user_id"] == user_id]
        if user_history.empty:
            logger.info("User %s has no history. Falling back to popularity recommender", user_id)
            return self._cold_start_recommendations(top_k)

        popularity_scores = self.popularity_recommender.score()
        content_scores = self.content_recommender.score_user(user_id)
        category_scores = self._category_scores_for_user(user_id)

        combined = self.hybrid_recommender.combine_scores(
            popularity_scores=popularity_scores,
            content_scores=content_scores,
            category_scores=category_scores,
        )

        seen_products = self.user_seen_products.get(user_id, set())
        filtered = combined[~combined["product_id"].isin(seen_products)].copy()

        filtered = filtered.merge(
            self.products_df[["product_id", "name", "category", "brand", "price"]],
            on="product_id",
            how="left",
        )

        filtered["recommendation_reason"] = "hybrid_content_popularity_category"
        filtered = filtered.head(top_k).reset_index(drop=True)
        filtered = filtered.rename(columns={"final_score": "score"})

        return filtered[
            ["product_id", "name", "category", "brand", "price", "score", "recommendation_reason"]
        ]


def load_processed_data(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users_df = pd.read_csv(config.paths.processed_data_dir / "users_processed.csv")
    products_df = pd.read_csv(config.paths.processed_data_dir / "products_processed.csv")
    interactions_df = pd.read_csv(config.paths.processed_data_dir / "interactions_processed.csv")
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    return users_df, products_df, interactions_df


def build_and_save_service(config: AppConfig) -> None:
    ensure_directories(
        [
            config.paths.raw_data_dir,
            config.paths.processed_data_dir,
            config.paths.artifacts_dir,
            config.paths.models_dir,
        ]
    )

    users_df, products_df, interactions_df = load_processed_data(config)

    service = RecommendationService(
        users_df=users_df,
        products_df=products_df,
        interactions_df=interactions_df,
        config=config,
    )

    artifact = {
        "users_df": users_df,
        "products_df": products_df,
        "interactions_df": interactions_df,
        "config": config,
    }

    joblib.dump(artifact, config.paths.models_dir / "recommendation_service.joblib")
    logger.info("Saved service artifact to %s", config.paths.models_dir / "recommendation_service.joblib")

    _ = service.recommend_for_user(users_df["user_id"].iloc[0], top_k=5)
    logger.info("Model build sanity check completed")


def load_service(config: AppConfig) -> RecommendationService:
    artifact_path = config.paths.models_dir / "recommendation_service.joblib"
    artifact = joblib.load(artifact_path)

    service = RecommendationService(
        users_df=artifact["users_df"],
        products_df=artifact["products_df"],
        interactions_df=artifact["interactions_df"],
        config=artifact["config"],
    )
    return service
