from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.config import PATHS


REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "PULocationID",
    "DOLocationID",
]


def assert_required_columns(columns: list[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    spark = (
        SparkSession.builder.appName("rl-data-platform-transform")
        .master("local[*]")
        .getOrCreate()
    )

    bronze_df = spark.read.parquet(str(PATHS.bronze))
    assert_required_columns(bronze_df.columns)

    silver_df = (
        bronze_df.filter(F.col("trip_distance") > 0)
        .filter(F.col("fare_amount") > 0)
        .withColumn("trip_minutes", (F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long")) / 60.0)
        .withColumn("pickup_hour", F.hour(F.col("tpep_pickup_datetime")))
        .withColumn("speed_mph_proxy", F.col("trip_distance") / (F.col("trip_minutes") / 60.0))
        .filter((F.col("trip_minutes") > 1) & (F.col("trip_minutes") < 240))
        .filter((F.col("speed_mph_proxy") > 1) & (F.col("speed_mph_proxy") < 80))
        .select(
            "tpep_pickup_datetime",
            "pickup_hour",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "trip_minutes",
            "fare_amount",
            "speed_mph_proxy",
        )
    )

    gold_df = (
        silver_df.groupBy("pickup_hour", "PULocationID")
        .agg(
            F.count("*").alias("trip_count"),
            F.avg("trip_minutes").alias("avg_trip_minutes"),
            F.avg("fare_amount").alias("avg_fare_amount"),
            F.avg("speed_mph_proxy").alias("avg_speed_mph_proxy"),
        )
        .withColumn("demand_bucket", F.when(F.col("trip_count") >= 500, F.lit("high")).otherwise(F.lit("normal")))
    )

    PATHS.silver.parent.mkdir(parents=True, exist_ok=True)
    PATHS.gold.parent.mkdir(parents=True, exist_ok=True)
    silver_df.write.mode("overwrite").parquet(str(PATHS.silver))
    gold_df.write.mode("overwrite").parquet(str(PATHS.gold))

    print(f"[transform] silver rows={silver_df.count()} -> {PATHS.silver}")
    print(f"[transform] gold rows={gold_df.count()} -> {PATHS.gold}")
    spark.stop()


if __name__ == "__main__":
    main()
