"""This module contains the code for Task 5 of the assignment.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import nltk
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import StringType

from task_5.sentiment_analysis import SentimentAnalysis
from utils.util import deduplicate_data

nltk.download("sentiwordnet")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

__author__ = "Cheng Yi Xing"


def run(data_dir: Optional[str | os.PathLike | Path]):
    """Run the Task 5 of the Airline Twitter Sentiment Analysis.

    :param data_dir: The directory path where the data is stored.
    :type data_dir: Optional[str  |  os.PathLike  |  Path]
    """
    print("Running Task 2")
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    spark = SparkSession.builder.appName(
        "Airline Twitter Sentiment Analysis"
    ).getOrCreate()

    deduplicate_data(
        data_dir=data_dir,
        spark=spark,
        output_dir=os.path.join(data_dir, "processed", "task_5"),
        columns_to_drop=[
            "_region",
            "_city",
            "_tainted",
            "most_common_user_timezone",
        ],
    )

    df = spark.read.csv(
        os.path.join(data_dir, "processed", "task_5"),
        header=True,
        inferSchema=True,
    )

    sentiment_class = SentimentAnalysis()
    df = df.withColumn(
        "sentiment", udf(sentiment_class.get_sentiment, StringType())(col("text"))
    )

    df = df.withColumn(
        "sentiment_without_stop_words",
        udf(sentiment_class.get_sentiment_stop_words_removed, StringType())(
            col("text")
        ),
    )

    df = df.withColumn(
        "get_sentiment_inverse_if_negative",
        udf(sentiment_class.get_sentiment_inverse_if_negative, StringType())(
            col("text")
        ),
    )

    df = df.withColumn(
        "get_sentiment_higher_threshold",
        udf(sentiment_class.get_sentiment_higher_threshold, StringType())(col("text")),
    )

    df = df.withColumn(
        "get_sentiment_lower_threshold",
        udf(sentiment_class.get_sentiment_lower_threshold, StringType())(col("text")),
    )

    df = df.withColumn(
        "get_sentiment_vader",
        udf(sentiment_class.get_sentiment_vader, StringType())(col("text")),
    )

    to_analyse_columns = [
        "sentiment",
        "sentiment_without_stop_words",
        "get_sentiment_inverse_if_negative",
        "get_sentiment_higher_threshold",
        "get_sentiment_lower_threshold",
        "get_sentiment_vader",
    ]

    df = df[df["airline_sentiment_gold"].isNotNull()]

    total_rows = df.count()

    for column in to_analyse_columns:
        is_matching_column = f"{column}_is_matching"
        print(f"Calculating accuracy for {column}")
        df = df.withColumn(
            is_matching_column,
            when(col(column) == col("airline_sentiment_gold"), 1).otherwise(0),
        )
        matches = df.filter(col(is_matching_column) == 1).count()

        accuracy = (matches / total_rows) * 100

        print(f"Total matches for {column} is {matches}/{total_rows}")
        print(f"Accuracy for {column} is {accuracy}%")

    df.repartition(1).write.csv(
        "./Twitter_Airline Dataset/sentiment", header=True, mode="overwrite", quote='"'
    )

    print("Task 2 completed")

    spark.stop()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Task 5")
    args.add_argument("--data_dir", type=str, help="Path to the data directory")
    run(**vars(args.parse_args()))
