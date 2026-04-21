from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.config import PATHS, SETTINGS


SUPPORTED_SOURCES = {"nyc_tlc", "inventory_synth"}


def build_inventory_synth_df(spark: SparkSession):
    # Generates a normalized bronze schema compatible with transform_spark.py.
    n_rows = 20000
    df = spark.range(0, n_rows).withColumnRenamed("id", "row_id")
    return (
        df.withColumn(
            "tpep_pickup_datetime",
            F.to_timestamp(
                F.from_unixtime(F.lit(1704067200) + (F.col("row_id") % 43200) * 60)
            ),
        )
        .withColumn(
            "tpep_dropoff_datetime",
            F.to_timestamp(
                F.from_unixtime(
                    F.lit(1704067200)
                    + ((F.col("row_id") % 43200) + 12 + (F.col("row_id") % 45)) * 60
                )
            ),
        )
        .withColumn("trip_distance", (F.rand(seed=42) * 7 + 0.3).cast("double"))
        .withColumn("fare_amount", (F.rand(seed=7) * 40 + 4).cast("double"))
        .withColumn("PULocationID", ((F.col("row_id") % 200) + 1).cast("int"))
        .withColumn("DOLocationID", (((F.col("row_id") + 17) % 200) + 1).cast("int"))
        .drop("row_id")
    )


def main() -> None:
    spark = (
        SparkSession.builder.appName("rl-data-platform-ingest")
        .master("local[*]")
        .getOrCreate()
    )

    if SETTINGS.data_source not in SUPPORTED_SOURCES:
        raise ValueError(
            f"Unsupported DATA_SOURCE='{SETTINGS.data_source}'. "
            f"Use one of {sorted(SUPPORTED_SOURCES)}."
        )

    if SETTINGS.data_source == "nyc_tlc":
        print(f"[ingest] DATA_SOURCE=nyc_tlc reading source parquet: {SETTINGS.source_url}")
        raw_df = spark.read.parquet(SETTINGS.source_url)
    else:
        print("[ingest] DATA_SOURCE=inventory_synth generating synthetic operations dataset")
        raw_df = build_inventory_synth_df(spark)

    bronze_path = SETTINGS.bronze_path_override or str(PATHS.bronze)
    PATHS.bronze.parent.mkdir(parents=True, exist_ok=True)
    raw_df.write.mode("overwrite").parquet(bronze_path)
    print(f"[ingest] bronze rows={raw_df.count()} -> {bronze_path}")

    spark.stop()


if __name__ == "__main__":
    main()
