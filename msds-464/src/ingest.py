from pathlib import Path
import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/ops_events.csv")


def generate_synthetic_ops_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    demand = rng.poisson(20, size=n_rows)
    inventory = rng.integers(5, 120, size=n_rows)
    lead_time = rng.integers(1, 8, size=n_rows)
    holding_cost = rng.uniform(0.5, 2.0, size=n_rows).round(3)
    shortage_cost = rng.uniform(2.0, 8.0, size=n_rows).round(3)

    df = pd.DataFrame(
        {
            "timestep": np.arange(n_rows),
            "demand": demand,
            "inventory": inventory,
            "lead_time_days": lead_time,
            "holding_cost": holding_cost,
            "shortage_cost": shortage_cost,
        }
    )
    return df


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_ops_data()
    df.to_csv(RAW_PATH, index=False)
    print(f"[ingest] wrote {len(df)} rows to {RAW_PATH}")


if __name__ == "__main__":
    main()
