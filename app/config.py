from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataGenerationConfig:
    n_users: int = 1000
    n_products: int = 500
    n_interactions: int = 10000
    random_seed: int = 42


@dataclass
class HybridWeightsConfig:
    popularity_weight: float = 0.20
    content_weight: float = 0.60
    category_weight: float = 0.20


@dataclass
class EvaluationConfig:
    test_ratio: float = 0.20
    default_k: int = 10
    min_user_train_interactions: int = 3
    min_user_test_interactions: int = 1


@dataclass
class PathConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    models_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.artifacts_dir = self.project_root / "artifacts"
        self.models_dir = self.artifacts_dir / "models"


@dataclass
class AppConfig:
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    hybrid_weights: HybridWeightsConfig = field(default_factory=HybridWeightsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def get_config() -> AppConfig:
    return AppConfig()
