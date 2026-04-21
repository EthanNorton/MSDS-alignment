from pathlib import Path
import pandas as pd


def test_gold_data_has_expected_columns() -> None:
    gold_path = Path("data/gold/decision_signals")
    if not gold_path.exists():
        # Smoke guard so test remains useful before first pipeline run.
        return

    df = pd.read_parquet(gold_path)
    expected = {"pickup_hour", "PULocationID", "trip_count", "avg_trip_minutes"}
    assert expected.issubset(df.columns)
