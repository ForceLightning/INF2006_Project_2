"""This module contains utility functions for the project.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType
)
from tqdm.auto import tqdm

DATA_DIR = "data"
SENTIMENT_DATA_DIR = "Twitter_Airline Dataset/sentiment"

def load_data(
    data_dir: str | os.PathLike = DATA_DIR, spark_session: Optional[SparkSession] = None
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
            StructField("_created_at", StringType(), False),
            StructField("_golden", BooleanType(), False),
            StructField("_id", IntegerType(), False),
            StructField("_missed", BooleanType(), True),
            StructField("_started_at", StringType(), False),
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
            StructField("tweet_created", StringType(), False),
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


def deduplicate_data(
    data_dir: str | os.PathLike | Path = DATA_DIR,
    spark: Optional[SparkSession] = None,
    output_dir: Optional[str | os.PathLike | Path] = None,
    columns_to_drop: Optional[list[str]] = None,
    remove_newline: bool = False,
) -> None:
    """De-duplicates the data, processes timestamps, and writes the processed data to a CSV file.

    :param data_dir: Unprocessed data directory, defaults to DATA_DIR
    :type data_dir: str | os.PathLike | Path, optional
    :param spark: Spark session object, defaults to None
    :type spark: Optional[SparkSession], optional
    :param output_dir: Output directory for processed csv, defaults to None
    :type output_dir: Optional[str  |  os.PathLike  |  Path], optional
    :param columns_to_drop: Columns to drop from the output, defaults to None
    :type columns_to_drop: Optional[list[str]], optional
    """
    if not columns_to_drop:
        columns_to_drop = [
            "airline_sentiment_gold",
            "negativereason_gold",
            "negativereason",
            "_region",
            "_city",
            "_tainted",
            "most_common_user_timezone",
        ]

    if not spark:
        spark = SparkSession.builder.appName(
            "Airline Twitter Sentiment Analysis"
        ).getOrCreate()

    if not output_dir:
        output_dir = os.path.join(data_dir, "processed")
    df = load_data(data_dir, spark)

    # Removing duplicates
    df_deduplicated = df.dropDuplicates(
        ["tweet_id", "_id", "text", "tweet_created", "name", "_created_at"]
    )

    print("Number of rows before deduplication: ", df.count())
    print("Number of rows after deduplication: ", df_deduplicated.count())

    # Converting to timestamps
    df_deduplicated = df_deduplicated.withColumn(
        "tweet_created", F.to_timestamp("tweet_created", "d/M/yyyy H:mm")
    )

    # Removal of the seconds from the timestamps
    df_deduplicated = df_deduplicated.withColumn(
        "_created_at",
        F.when(
            F.length(F.col("_created_at")) == 18,
            F.substring(F.col("_created_at"), 1, 15),
        ).otherwise(F.col("_created_at")),
    )

    df_deduplicated = df_deduplicated.withColumn(
        "_started_at",
        F.when(
            F.length(F.col("_started_at")) == 18,
            F.substring(F.col("_started_at"), 1, 15),
        ).otherwise(F.col("_started_at")),
    )

    # Convert the processed timestamp string to a timestamp type
    df_deduplicated = df_deduplicated.withColumn(
        "_created_at", F.to_timestamp("_created_at", "M/d/yyyy H:mm")
    )

    df_deduplicated = df_deduplicated.withColumn(
        "_started_at", F.to_timestamp("_started_at", "M/d/yyyy H:mm")
    )

    # Remove \n and \r \t from the text column
    df_deduplicated = df_deduplicated.withColumn(
        "text", F.regexp_replace("text", "\n|\r|\t", "")
    )

    # Replace all double quotes with single quotes
    df_deduplicated = df_deduplicated.withColumn(
        "text", F.regexp_replace("text", '"', "'")
    )

    # Filling missing values in the '_missed' column with False
    df_deduplicated = df_deduplicated.withColumn(
        "_missed",
        F.when(df_deduplicated["_missed"].isNull(), False).otherwise(
            df_deduplicated["_missed"].cast("boolean")
        ),
    )

    # Filling missing values in the 'retweet_count' column with 0
    df_deduplicated = df_deduplicated.withColumn(
        "retweet_count",
        F.when(df_deduplicated["retweet_count"].isNull(), 0).otherwise(
            df_deduplicated["retweet_count"]
        ),
    )

    # Update 'airline_sentiment' column with values from 'airline_sentiment_gold' where 'airline_sentiment_gold' is not null
    df_deduplicated = df_deduplicated.withColumn(
        "airline_sentiment",
        F.when(
            df_deduplicated["airline_sentiment_gold"].isNotNull(),
            df_deduplicated["airline_sentiment_gold"],
        ).otherwise(df_deduplicated["airline_sentiment"]),
    )

    # Split the 'negativereason_gold' column by '&' and create new columns
    df_deduplicated = df_deduplicated.withColumn(
        "negativereason_split", F.split("negativereason_gold", "\n")
    )

    # Extract values from the split column into separate columns
    df_deduplicated = df_deduplicated.withColumn(
        "negativereason1", df_deduplicated.negativereason_split.getItem(0)
    ).withColumn("negativereason2", df_deduplicated.negativereason_split.getItem(1))

    # If negativereason1 is null, use the value in negativereason
    df_deduplicated = df_deduplicated.withColumn(
        "negativereason1",
        F.when(
            df_deduplicated["negativereason1"].isNull(),
            df_deduplicated["negativereason"],
        ).otherwise(df_deduplicated["negativereason1"]),
    )

    # Drop the intermediate split column
    df_deduplicated = df_deduplicated.drop("negativereason_split")

    # Filling missing values in 'negativereason1' and 'negativereason2' with 'Unknown'
    df_deduplicated = df_deduplicated.withColumn(
        "negativereason1",
        F.when(df_deduplicated["negativereason1"].isNull(), "Unknown").otherwise(
            df_deduplicated["negativereason1"]
        ),
    )
    df_deduplicated = df_deduplicated.withColumn(
        "negativereason2",
        F.when(df_deduplicated["negativereason2"].isNull(), "Unknown").otherwise(
            df_deduplicated["negativereason2"]
        ),
    )

    # Group by country and user_timezone, count occurrences, and order by count in descending order
    grouped_df = (
        df_deduplicated.groupBy("_country", "user_timezone")
        .agg(F.count("*").alias("count"))
        .orderBy("_country", F.desc("count"))
    )

    # Get the most common user_timezone for each country
    most_common_user_timezone = grouped_df.groupBy("_country").agg(
        F.collect_list("user_timezone").getItem(0).alias("most_common_user_timezone")
    )

    # Left join to fill missing values in 'user_timezone' with the most common value for the corresponding country
    df_deduplicated = df_deduplicated.join(
        most_common_user_timezone, on="_country", how="left"
    ).withColumn(
        "user_timezone",
        F.coalesce(
            F.col("user_timezone"), F.col("most_common_user_timezone"), F.lit("Unknown")
        ),
    )

    # Drop the intermediate 'grouped_df' DataFrame
    grouped_df.unpersist()

    # Fill missing values in 'user_timezone' with 'Unknown'
    df_deduplicated = df_deduplicated.withColumn(
        "user_timezone",
        F.when(F.col("user_timezone").isNull(), "Unknown").otherwise(
            F.col("user_timezone")
        ),
    )

    # Fill missing values in '_country' with 'Unknown'
    df_deduplicated = df_deduplicated.withColumn(
        "_country",
        F.when(F.col("_country").isNull(), "Unknown").otherwise(F.col("_country")),
    )

    # Drop inconsistent data columns and columns that have been split
    # _tainted is dropped because all the values are False after deduplication
    df_deduplicated = df_deduplicated.drop(*columns_to_drop)

    df_deduplicated = df_deduplicated.dropna(
        subset=["tweet_location", "tweet_coord"], how="all"
    )

    if remove_newline:
        df_deduplicated = df_deduplicated.withColumn(
            "text", F.regexp_replace("text", "\n|\r|\t", " ")
        ).withColumn(
            "negativereason_gold",
            F.regexp_replace("negativereason_gold", "\n|\r|\t", ","),
        )

    # Show the resulting DataFrame
    # df_deduplicated.show()

    # Write the DataFrame to a CSV file
    # df_deduplicated.write.csv('./Twitter_Airline Dataset/processed', header=True, mode='overwrite')

    df_deduplicated.repartition(1).write.csv(
        output_dir, header=True, mode="overwrite", quote='"'
    )


def load_sentiment_data(
    data_dir: str | os.PathLike = DATA_DIR, spark_session: Optional[SparkSession] = None
) -> pyspark.sql.DataFrame:
    """Loads data from the data directory into a Spark DataFrame.

    :param data_dir: Data directory, defaults to DATA_DIR
    :type data_dir: str | os.PathLike, optional
    :param spark_session: Spark session if it has already been instantiated, defaults to None
    :type spark_session: SparkSession, optional
    :return: Combined DataFrame of all CSV files in the data directory
    :rtype: pyspark.sql.DataFrame
    """

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
            StructField("_ip", StringType(), False),
            StructField("airline_sentiment", StringType(), False),
            StructField("negativereason", StringType(), True),
            StructField("airline", StringType(), False),
            StructField("airline_sentiment_gold", StringType(), True),
            StructField("name", StringType(), False),
            StructField("negativereason_gold", StringType(), True),
            StructField("retweet_count", IntegerType(), False),
            StructField("text", StringType(), False),
            StructField("tweet_created", TimestampType(), False),
            StructField("tweet_id", FloatType(), False),
            StructField("user_timezone", StringType(), True),
            StructField("negativereason1", StringType(), True),
            StructField("negativereason2", StringType(), True),
            StructField("sentiment", StringType(), True),
        ]
    )

    # Check the csv files in the data directory
    file = [file for file in os.listdir(data_dir) if file.endswith(".csv")][0]

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

    return df


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Process the data")
    args.add_argument("--data_dir", type=str, default=DATA_DIR)
    args.add_argument("--output_dir", type=str, default=None)
    args.add_argument("--remove_newline", default=False, action="store_true")
    args.add_argument("--columns_to_drop", nargs="+", type=list[str], default=None)
    deduplicate_data(**vars(args.parse_args()))
