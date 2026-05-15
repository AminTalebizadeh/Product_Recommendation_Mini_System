from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    def __init__(self, products_df: pd.DataFrame) -> None:
        required_columns = {"product_id", "normalized_popularity"}

        missing = required_columns.difference(products_df.columns)

        if missing:
            raise ValueError(
                f"Missing columns for PopularityRecommender: {missing}"
            )

        self.products_df = (
            products_df[
                ["product_id", "normalized_popularity"]
            ]
            .drop_duplicates(subset=["product_id"])
            .copy()
        )

    def score(self, top_k: int | None = None) -> pd.DataFrame:
        result = self.products_df.rename(
            columns={
                "normalized_popularity": "popularity_score"
            }
        ).copy()

        result["popularity_score"] = (
            result["popularity_score"]
            .fillna(0.0)
            .astype(float)
        )

        result = result.sort_values(
            by="popularity_score",
            ascending=False,
        ).reset_index(drop=True)

        if top_k is not None:
            result = result.head(top_k)

        return result