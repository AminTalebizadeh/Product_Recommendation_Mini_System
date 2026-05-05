from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.logger import get_logger
from app.utils import ensure_directories, set_random_seed

logger = get_logger(__name__)


CATEGORY_TAGS = {
    "Electronics": ["wireless", "smart", "portable", "bluetooth", "gaming", "usb"],
    "Home": ["kitchen", "decor", "storage", "comfort", "cleaning", "modern"],
    "Books": ["fiction", "business", "learning", "history", "science", "productivity"],
    "Clothing": ["casual", "sport", "cotton", "fashion", "winter", "summer"],
    "Beauty": ["skincare", "organic", "makeup", "haircare", "wellness", "daily"],
    "Sports": ["fitness", "training", "outdoor", "cardio", "strength", "performance"],
}

CATEGORY_PRODUCT_TEMPLATES = {
    "Electronics": [
        "Wireless Earbuds",
        "Gaming Mouse",
        "Mechanical Keyboard",
        "Portable Speaker",
        "Smart Watch",
        "USB-C Hub",
    ],
    "Home": [
        "Storage Basket",
        "Ceramic Mug",
        "Table Lamp",
        "Throw Blanket",
        "Wall Shelf",
        "Kitchen Organizer",
    ],
    "Books": [
        "Startup Strategy Guide",
        "Deep Learning Handbook",
        "Productivity Playbook",
        "Modern History Notes",
        "Science Essentials",
        "Fiction Collection",
    ],
    "Clothing": [
        "Running Hoodie",
        "Classic T-Shirt",
        "Winter Jacket",
        "Casual Sneakers",
        "Training Shorts",
        "Denim Shirt",
    ],
    "Beauty": [
        "Daily Face Wash",
        "Hydrating Serum",
        "Hair Repair Mask",
        "SPF Moisturizer",
        "Lip Care Set",
        "Body Lotion",
    ],
    "Sports": [
        "Yoga Mat",
        "Resistance Bands",
        "Running Bottle",
        "Fitness Tracker Strap",
        "Jump Rope",
        "Training Gloves",
    ],
}

BRANDS = [
    "TechNova",
    "UrbanLeaf",
    "HomeCraft",
    "PeakForm",
    "BrightCart",
    "MonoStudio",
    "PureBlend",
    "NorthAxis",
]

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS = ["M", "F"]
PRICE_SENSITIVITY = ["low", "medium", "high"]
ACTIVITY_LEVELS = ["low", "medium", "high"]

EVENT_WEIGHTS = {
    "view": 1.0,
    "add_to_cart": 2.0,
    "purchase": 3.0,
}


@dataclass
class GeneratedData:
    users: pd.DataFrame
    products: pd.DataFrame
    interactions: pd.DataFrame


def _random_price_for_category(category: str) -> float:
    if category == "Electronics":
        return round(random.uniform(30, 450), 2)
    if category == "Home":
        return round(random.uniform(10, 180), 2)
    if category == "Books":
        return round(random.uniform(8, 60), 2)
    if category == "Clothing":
        return round(random.uniform(15, 200), 2)
    if category == "Beauty":
        return round(random.uniform(7, 120), 2)
    if category == "Sports":
        return round(random.uniform(12, 220), 2)
    return round(random.uniform(10, 100), 2)


def _sample_preferred_categories() -> list[str]:
    categories = list(CATEGORY_TAGS.keys())
    count = random.choice([1, 2, 2, 3])
    return random.sample(categories, count)


def generate_users(n_users: int) -> pd.DataFrame:
    rows = []
    for idx in range(1, n_users + 1):
        preferred_categories = _sample_preferred_categories()
        rows.append(
            {
                "user_id": f"U{idx:04d}",
                "age_group": random.choice(AGE_GROUPS),
                "gender": random.choice(GENDERS),
                "price_sensitivity": random.choices(
                    PRICE_SENSITIVITY, weights=[0.2, 0.55, 0.25], k=1
                )[0],
                "activity_level": random.choices(
                    ACTIVITY_LEVELS, weights=[0.25, 0.5, 0.25], k=1
                )[0],
                "preferred_categories": "|".join(preferred_categories),
            }
        )
    return pd.DataFrame(rows)


def generate_products(n_products: int) -> pd.DataFrame:
    rows = []
    categories = list(CATEGORY_TAGS.keys())

    for idx in range(1, n_products + 1):
        category = random.choice(categories)
        template_name = random.choice(CATEGORY_PRODUCT_TEMPLATES[category])
        brand = random.choice(BRANDS)
        tags = random.sample(CATEGORY_TAGS[category], k=3)
        name = f"{brand} {template_name}"

        rows.append(
            {
                "product_id": f"P{idx:04d}",
                "name": name,
                "category": category,
                "brand": brand,
                "price": _random_price_for_category(category),
                "tags": "|".join(tags),
            }
        )

    return pd.DataFrame(rows)


def _event_type_for_user_activity(activity_level: str) -> str:
    if activity_level == "high":
        return random.choices(
            ["view", "add_to_cart", "purchase"],
            weights=[0.58, 0.24, 0.18],
            k=1,
        )[0]
    if activity_level == "medium":
        return random.choices(
            ["view", "add_to_cart", "purchase"],
            weights=[0.68, 0.20, 0.12],
            k=1,
        )[0]
    return random.choices(
        ["view", "add_to_cart", "purchase"],
        weights=[0.78, 0.16, 0.06],
        k=1,
    )[0]


def _price_affinity_multiplier(price: float, sensitivity: str) -> float:
    if sensitivity == "high":
        if price < 40:
            return 1.20
        if price < 120:
            return 1.00
        return 0.75
    if sensitivity == "medium":
        if price < 150:
            return 1.05
        return 0.95
    if price > 120:
        return 1.10
    return 0.95


def _activity_interaction_count(activity_level: str) -> tuple[int, int]:
    if activity_level == "high":
        return 14, 24
    if activity_level == "medium":
        return 8, 16
    return 3, 8


def generate_interactions(
    users_df: pd.DataFrame,
    products_df: pd.DataFrame,
    target_total_interactions: int,
) -> pd.DataFrame:
    interactions = []
    product_category_map = products_df.set_index("product_id")["category"].to_dict()
    products_records = products_df.to_dict("records")
    now = datetime.now()

    per_user_counts = []
    for _, user in users_df.iterrows():
        low, high = _activity_interaction_count(user["activity_level"])
        per_user_counts.append(random.randint(low, high))

    current_total = sum(per_user_counts)
    scale = target_total_interactions / max(current_total, 1)
    per_user_counts = [max(1, int(count * scale)) for count in per_user_counts]

    for user_idx, (_, user) in enumerate(users_df.iterrows()):
        user_id = user["user_id"]
        preferred_categories = user["preferred_categories"].split("|")
        activity_level = user["activity_level"]
        price_sensitivity = user["price_sensitivity"]
        interaction_count = per_user_counts[user_idx]

        scored_products = []
        for product in products_records:
            score = 1.0
            if product["category"] in preferred_categories:
                score *= 3.0
            score *= _price_affinity_multiplier(float(product["price"]), price_sensitivity)
            score *= 1.0 + np.random.uniform(-0.1, 0.1)
            scored_products.append(max(score, 0.01))

        scored_products = np.array(scored_products, dtype=float)
        probabilities = scored_products / scored_products.sum()

        chosen_indices = np.random.choice(
            len(products_records),
            size=interaction_count,
            replace=True,
            p=probabilities,
        )

        for selected_idx in chosen_indices:
            product = products_records[int(selected_idx)]
            category = product_category_map[product["product_id"]]
            event_type = _event_type_for_user_activity(activity_level)
            event_weight = EVENT_WEIGHTS[event_type]
            timestamp = now - timedelta(days=random.randint(0, 59), hours=random.randint(0, 23))

            interactions.append(
                {
                    "user_id": user_id,
                    "product_id": product["product_id"],
                    "event_type": event_type,
                    "event_weight": event_weight,
                    "category": category,
                    "timestamp": timestamp.isoformat(timespec="seconds"),
                }
            )

    interactions_df = pd.DataFrame(interactions)
    interactions_df = interactions_df.sort_values("timestamp").reset_index(drop=True)
    return interactions_df


def build_dataset(config: AppConfig) -> GeneratedData:
    set_random_seed(config.data_generation.random_seed)
    users_df = generate_users(config.data_generation.n_users)
    products_df = generate_products(config.data_generation.n_products)
    interactions_df = generate_interactions(
        users_df=users_df,
        products_df=products_df,
        target_total_interactions=config.data_generation.n_interactions,
    )
    return GeneratedData(users=users_df, products=products_df, interactions=interactions_df)


def save_generated_data(dataset: GeneratedData, config: AppConfig) -> None:
    ensure_directories(
        [
            config.paths.raw_data_dir,
            config.paths.processed_data_dir,
            config.paths.artifacts_dir,
            config.paths.models_dir,
        ]
    )

    dataset.users.to_csv(config.paths.raw_data_dir / "users.csv", index=False)
    dataset.products.to_csv(config.paths.raw_data_dir / "products.csv", index=False)
    dataset.interactions.to_csv(config.paths.raw_data_dir / "interactions.csv", index=False)

    logger.info("Saved users to %s", config.paths.raw_data_dir / "users.csv")
    logger.info("Saved products to %s", config.paths.raw_data_dir / "products.csv")
    logger.info("Saved interactions to %s", config.paths.raw_data_dir / "interactions.csv")


def generate_and_save_data(config: AppConfig) -> None:
    logger.info("Generating synthetic recommendation dataset")
    dataset = build_dataset(config)
    save_generated_data(dataset, config)
    logger.info(
        "Data generation completed | users=%d | products=%d | interactions=%d",
        len(dataset.users),
        len(dataset.products),
        len(dataset.interactions),
    )
