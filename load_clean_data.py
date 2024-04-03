import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, when, regexp_replace, trim
from pyspark.sql.types import (
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
import os
from util import load_data

DATA_DIR = "data"
SAVED_DIR = "Twitter_Airline Dataset/processed"


def load_clean_data(
    data_dir: str | os.PathLike = DATA_DIR, spark_session: SparkSession = None
) -> pyspark.sql.DataFrame:
    """Loads data from the data directory into a Spark DataFrame and performs data cleaning operations."""

    # Check if the saved directory exists
    if os.path.exists(SAVED_DIR):
        return spark_session.read.csv(SAVED_DIR, header=True, inferSchema=True)

    df = load_data(data_dir=data_dir, spark_session=spark_session)

    # Removing duplicates based on the tweet and id
    df_deduplicated = df.dropDuplicates(['text', '_id'])

    # Remove \n and \r \t from the text column
    df_deduplicated = df_deduplicated.withColumn(
        "text", regexp_replace("text", "\n|\r|\t", ""))

    # Replace all double quotes with single quotes
    df_deduplicated = df_deduplicated.withColumn(
        "text", regexp_replace("text", "\"", "'"))

    # Filling missing values in the '_missed' column with False
    df_deduplicated = df_deduplicated.withColumn("_missed", when(
        df_deduplicated["_missed"].isNull(), False).otherwise(df_deduplicated["_missed"].cast("boolean")))

    # Filling missing values in the 'retweet_count' column with 0
    df_deduplicated = df_deduplicated.withColumn("retweet_count", when(
        df_deduplicated["retweet_count"].isNull(), 0).otherwise(df_deduplicated["retweet_count"]))

    # Update 'airline_sentiment' column with values from 'airline_sentiment_gold' where 'airline_sentiment_gold' is not null
    df_deduplicated = df_deduplicated.withColumn("airline_sentiment", when(df_deduplicated["airline_sentiment_gold"].isNotNull(
    ), df_deduplicated["airline_sentiment_gold"]).otherwise(df_deduplicated["airline_sentiment"]))

    # Split the 'negativereason_gold' column by '&' and create new columns
    df_deduplicated = df_deduplicated.withColumn(
        "negativereason_split", split("negativereason_gold", "\n"))

    # Extract values from the split column into separate columns
    df_deduplicated = df_deduplicated.withColumn("negativereason1", df_deduplicated.negativereason_split.getItem(0)) \
        .withColumn("negativereason2", df_deduplicated.negativereason_split.getItem(1))

    # If negativereason1 is null, use the value in negativereason
    df_deduplicated = df_deduplicated.withColumn("negativereason1", when(df_deduplicated["negativereason1"].isNull(
    ), df_deduplicated["negativereason"]).otherwise(df_deduplicated["negativereason1"]))

    # Drop the intermediate split column
    df_deduplicated = df_deduplicated.drop("negativereason_split")

    # Drop inconsistent data columns and columns that have been split
    df_deduplicated = df_deduplicated.drop(
        "airline_sentiment_gold", "negativereason_gold", "negativereason", "_region", "_city", "tweet_location", "tweet_coord")

    df_deduplicated.repartition(1).write.csv(
        "./Twitter_Airline Dataset/processed", header=True, mode='overwrite', quote='\"')

    return df_deduplicated
