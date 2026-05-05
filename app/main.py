from __future__ import annotations

import typer

from app.config import get_config
from app.data.generate_data import generate_and_save_data
from app.data.preprocess import preprocess_and_save
from app.evaluation.evaluator import evaluate_recommender
from app.logger import get_logger
from app.recommender.service import build_and_save_service, load_processed_data, load_service

app = typer.Typer(help="Product Recommendation Mini System CLI")
logger = get_logger(__name__)


@app.command("generate-data")
def generate_data(
    n_users: int = typer.Option(None, help="Number of users to generate"),
    n_products: int = typer.Option(None, help="Number of products to generate"),
    n_interactions: int = typer.Option(None, help="Target number of interactions to generate"),
) -> None:
    config = get_config()

    if n_users is not None:
        config.data_generation.n_users = n_users
    if n_products is not None:
        config.data_generation.n_products = n_products
    if n_interactions is not None:
        config.data_generation.n_interactions = n_interactions

    generate_and_save_data(config)
    typer.echo("Synthetic data generated successfully.")


@app.command("preprocess")
def preprocess() -> None:
    config = get_config()
    preprocess_and_save(config)
    typer.echo("Preprocessing completed successfully.")


@app.command("build-model")
def build_model() -> None:
    config = get_config()
    build_and_save_service(config)
    typer.echo("Recommendation service artifact built successfully.")


@app.command("evaluate")
def evaluate(
    k: int = typer.Option(None, help="Top-k cutoff for evaluation"),
) -> None:
    config = get_config()
    if k is None:
        k = config.evaluation.default_k

    metrics = evaluate_recommender(config=config, k=k)

    typer.echo("Evaluation completed.")
    typer.echo(f"Evaluated users: {metrics['evaluated_users']}")
    typer.echo(f"Precision@{metrics['k']}: {metrics['precision_at_k']}")
    typer.echo(f"Recall@{metrics['k']}: {metrics['recall_at_k']}")
    typer.echo(f"HitRate@{metrics['k']}: {metrics['hit_rate_at_k']}")


@app.command("recommend")
def recommend(
    user_id: str = typer.Option(..., help="User ID for recommendation"),
    top_k: int = typer.Option(5, help="Number of recommendations"),
) -> None:
    config = get_config()
    service = load_service(config)
    recommendations = service.recommend_for_user(user_id=user_id, top_k=top_k)

    if recommendations.empty:
        typer.echo("No recommendations found.")
        return

    typer.echo(f"Top {top_k} recommendations for {user_id}:")
    typer.echo(recommendations.to_string(index=False))


@app.command("list-users")
def list_users(
    limit: int = typer.Option(10, help="Number of users to display"),
) -> None:
    config = get_config()
    users_df, _, _ = load_processed_data(config)
    typer.echo(users_df[["user_id", "age_group", "gender", "activity_level"]].head(limit).to_string(index=False))


@app.command("list-products")
def list_products(
    limit: int = typer.Option(10, help="Number of products to display"),
) -> None:
    config = get_config()
    _, products_df, _ = load_processed_data(config)
    typer.echo(
        products_df[["product_id", "name", "category", "brand", "price"]]
        .head(limit)
        .to_string(index=False)
    )


if __name__ == "__main__":
    app()
