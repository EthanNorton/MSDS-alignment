from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    bronze: Path = Path("data/bronze/taxi_trips")
    silver: Path = Path("data/silver/state_features")
    gold: Path = Path("data/gold/decision_signals")
    artifacts: Path = Path("data/artifacts")


@dataclass(frozen=True)
class Settings:
    # Supported values: "nyc_tlc", "inventory_synth"
    data_source: str = os.getenv("DATA_SOURCE", "nyc_tlc").strip().lower()
    source_url: str = os.getenv(
        "SOURCE_URL",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
    )
    bronze_path_override: str | None = os.getenv("BRONZE_PATH")
    random_seed: int = 42


PATHS = Paths()
SETTINGS = Settings()
