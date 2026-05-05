from __future__ import annotations

import pandas as pd

from app.config import HybridWeightsConfig


class HybridRecommender:
    def __init__(self, weights: HybridWeightsConfig) -> None:
        self.weights = weights

    def combine_scores(
        self,
        popularity_scores: pd.DataFrame,
        content_scores: pd.DataFrame,
        category_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        result = popularity_scores.merge(content_scores, on="product_id", how="outer")
        result = result.merge(category_scores, on="product_id", how="outer")
        result = result.fillna(0.0)

        result["final_score"] = (
            self.weights.popularity_weight * result["popularity_score"]
            + self.weights.content_weight * result["content_score"]
            + self.weights.category_weight * result["category_score"]
        )

        return result.sort_values("final_score", ascending=False).reset_index(drop=True)
