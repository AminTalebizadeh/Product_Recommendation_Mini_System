from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    def __init__(self, products_df: pd.DataFrame) -> None:
        required_columns = {"product_id", "normalized_popularity"}
        missing = required_columns.difference(products_df.columns)
        if missing:
            raise ValueError(f"Missing columns for PopularityRecommender: {missing}")

        self.products_df = products_df.copy()

    def score(self) -> pd.DataFrame:
        result = self.products_df[["product_id", "normalized_popularity"]].copy()
        result = result.rename(columns={"normalized_popularity": "popularity_score"})
        return result
