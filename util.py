import os

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tqdm.auto import tqdm

DATA_DIR = "data"


def load_data(
    data_dir: str | os.PathLike = DATA_DIR, spark_session: SparkSession = None
) -> pyspark.sql.DataFrame:
    """Loads data from the data directory into a Spark DataFrame.

    :param data_dir: Data directory, defaults to DATA_DIR
    :type data_dir: str | os.PathLike, optional
    :param spark_session: Spark session if it has already been instantiated, defaults to None
    :type spark_session: SparkSession, optional
    :return: Combined DataFrame of all CSV files in the data directory
    :rtype: pyspark.sql.DataFrame
    """
    all_files = [
        file
        for file in os.listdir(data_dir)
        if file.endswith(".csv") and file != "combined_csv.csv"
    ]
    all_dfs = []
    spark = (
        spark_session
        if spark_session is not None
        else SparkSession.builder.appName(
            "Airline Twitter Sentiment Analysis"
        ).getOrCreate()
    )

    schema = StructType(
        [
            StructField("_unit_id", IntegerType(), False),
            StructField("_created_at", TimestampType(), False),
            StructField("_golden", BooleanType(), False),
            StructField("_id", IntegerType(), False),
            StructField("_missed", BooleanType(), True),
            StructField("_started_at", TimestampType(), False),
            StructField("_tainted", BooleanType(), False),
            StructField("_channel", StringType(), False),
            StructField("_trust", FloatType(), False),
            StructField("_worker_id", IntegerType(), False),
            StructField("_country", StringType(), True),
            StructField("_region", StringType(), True),
            StructField("_city", StringType(), True),
            StructField("_ip", StringType(), False),
            StructField("airline_sentiment", StringType(), False),
            StructField("negativereason", StringType(), True),
            StructField("airline", StringType(), False),
            StructField("airline_sentiment_gold", StringType(), True),
            StructField("name", StringType(), False),
            StructField("negativereason_gold", StringType(), True),
            StructField("retweet_count", IntegerType(), False),
            StructField("text", StringType(), False),
            StructField(
                "tweet_coord", StringType(), True
            ),  # This should really be an ArrayType(FloatType)
            StructField("tweet_created", TimestampType(), False),
            StructField("tweet_id", FloatType(), False),
            StructField("tweet_location", StringType(), True),
            StructField("user_timezone", StringType(), True),
        ]
    )

    iterator = tqdm(all_files, desc="Loading data", unit="file")
    for file in iterator:
        df = (
            spark.read.option("wholeFile", True)
            .option("multiLine", True)
            .option("header", True)
            .option("inferSchema", False)
            .option("dateFormat", "m/d/yyyy")
            .option("timestampFormat", "M/d/yyyy HH:mm:ss")
            .option("quote", '"')
            .option("escape", '"')
            .csv(os.path.join(data_dir, file), schema=schema)
        )
        all_dfs.append(df)
        iterator.write(f"Loaded {file} with {df.count()} rows")

    super_df = all_dfs[0]
    for df in all_dfs[1:]:
        super_df = super_df.union(df)

    return super_df
