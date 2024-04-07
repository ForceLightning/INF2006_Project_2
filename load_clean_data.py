import os
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from util import load_data

DATA_DIR = "data"
SAVED_DIR = DATA_DIR + "/processed"


def load_clean_data(
    data_dir: str | os.PathLike | None = DATA_DIR,
    spark_session: SparkSession | None = None,
    save_to_dir: str | os.PathLike = SAVED_DIR,
) -> pyspark.sql.DataFrame:
    """Loads data from the data directory into a Spark DataFrame and performs
    data cleaning operations.

    :param data_dir: Data directory that contains the raw data, defaults to DATA_DIR
    :type data_dir: str | os.PathLike, optional
    :param spark_session: Optional spark session if already instantiated, defaults to None
    :type spark_session: SparkSession, optional
    :param save_to_dir: Directory to save the cleaned data to, defaults to SAVED_DIR
    :type save_to_dir: str | os.PathLike | None, optional
    :return: Cleaned DataFrame
    :rtype: pyspark.sql.DataFrame
    :raises ValueError: If both data_dir and save_to_dir are not provided
    """

    if not data_dir and not save_to_dir:
        raise ValueError(
            "At least one of data_dir or save_to_dir must be provided")

    spark_session = (
        spark_session
        if spark_session
        else SparkSession.builder.appName(
            "Airline Twitter Sentiment Analysis"
        ).getOrCreate()
    )

    # Check if the saved directory exists
    if save_to_dir:
        if os.path.exists(save_to_dir):
            return spark_session.read.csv(save_to_dir, header=True, inferSchema=True)

    df = load_data(data_dir=data_dir, spark_session=spark_session)

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
        F.collect_list("user_timezone").getItem(
            0).alias("most_common_user_timezone")
    )

    # Left join to fill missing values in 'user_timezone' with the most common value for the corresponding country
    df_deduplicated = df_deduplicated.join(
        most_common_user_timezone, on="_country", how="left"
    ).withColumn(
        "user_timezone",
        F.coalesce(
            F.col("user_timezone"), F.col(
                "most_common_user_timezone"), F.lit("Unknown")
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
        F.when(F.col("_country").isNull(),
               "Unknown").otherwise(F.col("_country")),
    )

    # Drop inconsistent data columns and columns that have been split
    # _tainted is dropped because all the values are False after deduplication
    df_deduplicated = df_deduplicated.drop(
        "airline_sentiment_gold",
        "negativereason_gold",
        "negativereason",
        "_region",
        "_city",
        "_tainted",
        "most_common_user_timezone",
    )

    df_deduplicated = df_deduplicated.dropna(
        subset=["tweet_location", "tweet_coord"], how="all"
    )

    # Write the DataFrame to a CSV file
    df_deduplicated.repartition(1).write.csv(
        SAVED_DIR, header=True, mode="overwrite", quote='"'
    )

    return df_deduplicated


if __name__ == "__main__":
    load_clean_data()
