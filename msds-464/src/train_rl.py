from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.config import PATHS, SETTINGS


GOLD_PATH = PATHS.gold
TRAIN_LOG_PATH = PATHS.artifacts / "train_metrics.json"


def run_baseline_training(df: pd.DataFrame, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    episodes = 25
    rewards = []
    for episode in range(episodes):
        reward = float(
            1000
            + 0.2 * df["trip_count"].mean()
            - 0.5 * df["avg_trip_minutes"].mean()
            + rng.normal(0, 10)
        )
        rewards.append(reward)

    return {
        "seed": seed,
        "episodes": episodes,
        "avg_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "reward_trace": rewards,
    }


def main() -> None:
    if not GOLD_PATH.exists():
        raise FileNotFoundError("Gold dataset missing. Run src/transform_spark.py first.")

    df = pd.read_parquet(GOLD_PATH)
    metrics = run_baseline_training(df, seed=SETTINGS.random_seed)

    TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[train] wrote metrics to {TRAIN_LOG_PATH}")


if __name__ == "__main__":
    main()
