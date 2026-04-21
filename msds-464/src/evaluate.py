from pathlib import Path
import json
import pandas as pd
from src.config import PATHS


GOLD_PATH = PATHS.gold
TRAIN_LOG_PATH = PATHS.artifacts / "train_metrics.json"
EVAL_PATH = PATHS.artifacts / "evaluation_report.json"


def evaluate_policy(gold_df: pd.DataFrame, train_metrics: dict) -> dict:
    avg_minutes = float(gold_df["avg_trip_minutes"].mean())
    avg_trips = float(gold_df["trip_count"].mean())
    return {
        "avg_trip_minutes": avg_minutes,
        "avg_trip_count": avg_trips,
        "avg_reward": train_metrics["avg_reward"],
        "status": "pass" if train_metrics["avg_reward"] >= 975 else "needs_improvement",
    }


def main() -> None:
    if not GOLD_PATH.exists() or not TRAIN_LOG_PATH.exists():
        raise FileNotFoundError("Missing inputs. Run transform and train steps first.")

    gold_df = pd.read_parquet(GOLD_PATH)
    train_metrics = json.loads(TRAIN_LOG_PATH.read_text(encoding="utf-8"))

    report = evaluate_policy(gold_df, train_metrics)
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[evaluate] wrote report to {EVAL_PATH}")


if __name__ == "__main__":
    main()
