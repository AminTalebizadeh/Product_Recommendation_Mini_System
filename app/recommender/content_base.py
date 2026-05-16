from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils import normalize_scores


class ContentBasedRecommender:
    def __init__(
        self,
        products_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
    ) -> None:
        self.products_df = products_df.copy()
        self.interactions_df = interactions_df.copy()

        self.vectorizer = DictVectorizer(sparse=False)
        self.product_feature_matrix = None
        self.product_ids = self.products_df["product_id"].tolist()
        self.product_index = {product_id: idx for idx, product_id in enumerate(self.product_ids)}

        self._fit_product_features()

    def _price_bucket(self, price: float) -> str:
        if price < 25:
            return "low"
        if price < 100:
            return "medium"
        return "high"

    def _product_to_feature_dict(self, row: pd.Series) -> dict:
        feature_dict = {
            f"category={row['category']}": 1.0,
            f"brand={row['brand']}": 1.0,
            f"price_bucket={self._price_bucket(float(row['price']))}": 1.0,
        }
        for tag in str(row["tags"]).split("|"):
            feature_dict[f"tag={tag.strip()}"] = 1.0
        return feature_dict

    def _fit_product_features(self) -> None:
        product_feature_dicts = [
            self._product_to_feature_dict(row)
            for _, row in self.products_df.iterrows()
        ]
        self.product_feature_matrix = self.vectorizer.fit_transform(product_feature_dicts)

    def _build_user_profile_vector(self, user_id: str) -> np.ndarray:
        user_interactions = self.interactions_df[self.interactions_df["user_id"] == user_id]
        if user_interactions.empty:
            return np.zeros(self.product_feature_matrix.shape[1], dtype=float)

        weighted_vectors = []
        total_weight = 0.0

        for _, row in user_interactions.iterrows():
            product_id = row["product_id"]
            event_weight = float(row["event_weight"])
            idx = self.product_index.get(product_id)
            if idx is None:
                continue
            weighted_vectors.append(self.product_feature_matrix[idx] * event_weight)
            total_weight += event_weight

        if not weighted_vectors or total_weight <= 0.0:
            return np.zeros(self.product_feature_matrix.shape[1], dtype=float)

        user_vector = np.sum(weighted_vectors, axis=0) / total_weight
        return user_vector

    def score_user(self, user_id: str) -> pd.DataFrame:
        user_vector = self._build_user_profile_vector(user_id).reshape(1, -1)
        similarities = cosine_similarity(user_vector, self.product_feature_matrix)[0]
        similarities = normalize_scores(similarities)

        result = pd.DataFrame(
            {
                "product_id": self.product_ids,
                "content_score": similarities,
            }
        )
        return result
