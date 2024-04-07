"""This module contains utility functions for the project.
"""

import json
import os
import re

import pyproj
import pyspark
import pyspark.pandas as ps
import geopandas as gpd
from Levenshtein import ratio
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from shapely.geometry import Point
from tqdm.auto import tqdm

DATA_DIR = "data"
SENTIMENT_DATA_DIR = "Twitter_Airline Dataset/sentiment"

with open(os.path.join(DATA_DIR, "cities.json"), "r", encoding="utf-8") as f:
    cities_json = json.load(f)

with open(os.path.join(DATA_DIR, "countries.json"), "r", encoding="utf-8") as f:
    countries_json = json.load(f)

with open(os.path.join(DATA_DIR, "states.json"), "r", encoding="utf-8") as f:
    states_json = json.load(f)

with open(os.path.join(DATA_DIR, "cities.json"), "r", encoding="utf-8") as f:
    cities_json = json.load(f)

with open(os.path.join(DATA_DIR, "countries.json"), "r", encoding="utf-8") as f:
    countries_json = json.load(f)

with open(os.path.join(DATA_DIR, "states.json"), "r", encoding="utf-8") as f:
    states_json = json.load(f)


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


def match_location_with_country_by_city(
    location: str, cities: dict = cities_json, countries: dict = countries_json
):
    # Match the city name with the list of cities.
    country = None
    max_ratio = 0.0
    for c in cities:
        for word in location.split():
            current_ratio = ratio(word, c["city"])
            if current_ratio >= 0.8 and current_ratio > max_ratio:
                country = c["country_name"]
                break
    if country:
        country = [c["iso3"]
                   for c in countries if country["name"] == country][0]
    return country


def match_location_with_country_by_state(
    location: str, states: dict = states_json, countries: dict = countries_json
):
    country = None
    max_ratio = 0
    for s in states:
        for word in location.split():
            current_ratio = ratio(word, s["name"])
            if current_ratio >= 0.8 and current_ratio > max_ratio:
                country = s["country_name"]
                break
    if country:
        country = [c["iso3"]
                   for c in countries if country["name"] == country][0]
    return country


def match_location_with_country_by_country(
    location: str, countries: dict = countries_json
):
    country = None
    max_ratio = 0
    for c in countries:
        for word in location.split():
            # Test if the word is in the ISO3 code.
            if word.lower() == c["iso3"].lower():
                country = c["iso3"]
                break
            # Test if the word is in the country name.
            current_ratio = ratio(word, c["name"])
            if current_ratio >= 0.8 and current_ratio > max_ratio:
                country = c["iso3"]
                break

    return country


def load_sentiment_data(
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
    file = [
        file
        for file in os.listdir(data_dir)
        if file.endswith(".csv")
    ][0]

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
