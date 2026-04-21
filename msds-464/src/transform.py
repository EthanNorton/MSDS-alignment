from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/ops_events.csv")
PROCESSED_PATH = Path("data/processed/state_features.csv")

REQUIRED_COLUMNS = {
    "timestep",
    "demand",
    "inventory",
    "lead_time_days",
    "holding_cost",
    "shortage_cost",
}


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {sorted(missing)}")


def build_state_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["inventory_gap"] = features["inventory"] - features["demand"]
    features["cost_ratio"] = features["shortage_cost"] / features["holding_cost"]
    features["demand_rolling_mean_5"] = (
        features["demand"].rolling(window=5, min_periods=1).mean()
    )
    return features


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError("Raw data missing. Run src/ingest.py first.")

    df = pd.read_csv(RAW_PATH)
    validate_schema(df)
    features = build_state_features(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_PATH, index=False)
    print(f"[transform] wrote {len(features)} rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
