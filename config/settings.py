"""
config/settings.py — Central configuration for Aegis-Omni
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    input_size: int = 40          # number of features
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    sequence_length: int = 12     # 12 one-hour windows = 12h lookback
    num_ensemble_models: int = 3


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    val_split: float = 0.15
    test_split: float = 0.15
    target_fdr: float = 0.05      # False Discovery Rate target


@dataclass
class FeatherlessConfig:
    api_key: str = field(default_factory=lambda: os.getenv("FEATHERLESS_API_KEY", ""))
    base_url: str = "https://api.featherless.ai/v1"
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    timeout: int = 30


@dataclass
class DatabaseConfig:
    url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL",
        "postgresql://aegis:aegis_dev@localhost:5432/aegis_omni"
    ))


@dataclass
class DriftConfig:
    check_interval_minutes: int = 30
    ks_threshold: float = 0.05
    psi_threshold: float = 0.2
    window_size: int = 500


@dataclass
class FederatedConfig:
    num_rounds: int = 5
    num_hospitals: int = 3
    min_fit_clients: int = 2
    local_epochs: int = 3
    fraction_fit: float = 1.0


@dataclass
class AegisConfig:
    hospital_id: str = "HOSPITAL_001"
    device: str = "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    featherless: FeatherlessConfig = field(default_factory=FeatherlessConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    data_path: str = "data/sepsis.csv"
    checkpoint_dir: str = "checkpoints"
