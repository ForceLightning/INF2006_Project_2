"""This module performs the preprocessing step (after task 1) for task 3's main
MapReduce job.
"""

import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract
from scipy.spatial import cKDTree
from shapely.geometry import Point

from utils.util import deduplicate_data

DATA_DIR = "data"
PROCESSED_CSV_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "task_3")

__author__ = "Christopher Kok"


def ckdnearest(
    gd_a: gpd.GeoDataFrame, gd_b: gpd.GeoDataFrame, threshold: float
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Given two GeoDataFrames, this function finds the nearest point in gdB for each point in gdA.

    :param gd_a: GeoDataFrame containing the points to find the nearest point in gdB
    :type gd_a: gpd.GeoDataFrame
    :param gd_b: GeoDataFrame containing the points to find the nearest point in gdA
    :type gd_b: gpd.GeoDataFrame
    :param threshold: Maximum distance to consider
    :type threshold: float
    :return: Two GeoDataFrames containing the nearest points
    :rtype: Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    """
    n_a = np.array(list(gd_a.geometry.apply(lambda x: (x.x, x.y))))
    n_b = np.array(list(gd_b.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(n_b)
    print(
        np.isnan(n_a).any(),
        np.isnan(n_b).any(),
        np.any(np.isinf(n_a)),
        np.any(np.isinf(n_b)),
    )
    dist, idx = btree.query(n_a, k=1, distance_upper_bound=threshold)
    gd_b_nearest = gd_b.iloc[[i - 1 for i in idx]].reset_index(drop=True)
    gd_a = gd_a.reset_index(drop=True)
    gd_a["distance"] = dist
    return (
        gd_a[gd_a["distance"] <= threshold],
        gd_b_nearest[gd_a["distance"] <= threshold],
    )


def get_processed_csv_path(processed_csv_dir: str | os.PathLike | Path) -> str | None:
    """Given the processed CSV directory, this function returns the path of the processed CSV file.

    :param processed_csv_dir: Directory containing the processed CSV file
    :type processed_csv_dir: str | os.PathLike | Path
    :return: Path of the processed CSV file
    :rtype: str
    """
    if os.path.exists(processed_csv_dir):
        processed_csv = [
            os.path.join(processed_csv_dir, file)
            for file in os.listdir(processed_csv_dir)
            if file.endswith(".csv")
        ]
        if not processed_csv:
            return None
    else:
        os.makedirs(processed_csv_dir, exist_ok=True)
        return None
    return processed_csv[0]


def extract_countries_from_latlong(
    spark: SparkSession, df: DataFrame, countries: pd.DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Extracts the latitude and longitude from the tweet_coord and tweet_location columns,
    and finds the nearest country for each point.

    :param spark: Spark session object
    :type spark: SparkSession
    :param df: Main DataFrame
    :type df: DataFrame
    :param countries: Pandas DataFrame containing the countries data
    :type countries: pd.DataFrame
    :return: Two DataFrames containing the main DataFrame, and the DataFrame with a matched country.
    :rtype: tuple[DataFrame, DataFrame]
    """
    # Extract the latitude and longitude from the tweet_coord column assuming it
    # is in the format "latitude,longitude"
    #
    df = df.withColumn(
        "latitude",
        F.when(F.col("tweet_coord").isNotNull(), F.split(F.col("tweet_coord"), ",")[0])
        .otherwise(None)
        .cast("double"),
    ).withColumn(
        "longitude",
        F.when(F.col("tweet_coord").isNotNull(), F.split(F.col("tweet_coord"), ",")[1])
        .otherwise(None)
        .cast("double"),
    )

    # Extract the latitude and longitude with regex.
    pattern = r"([\+-]?\d+(?:\.\d+)?),\s?([\+-]?\d+(?:\.\d+)?)"
    df = df.withColumn(
        "latitude",
        F.when(
            F.col("tweet_coord").isNull() & F.col("tweet_location").isNotNull(),
            regexp_extract(F.col("tweet_location"), pattern, 1),
        )
        .otherwise(F.col("latitude"))
        .cast("double"),
    ).withColumn(
        "longitude",
        F.when(
            F.col("tweet_coord").isNull() & F.col("tweet_location").isNotNull(),
            regexp_extract(F.col("tweet_location"), pattern, 2),
        )
        .otherwise(F.col("longitude"))
        .cast("double"),
    )

    # ! Keep the index column for joining later
    indexed = df.withColumn("index", F.monotonically_increasing_id())
    latlong_df = indexed.filter(
        F.col("latitude").isNotNull() & F.col("longitude").isNotNull()
    )
    latlong_pddf = latlong_df.select("index", "latitude", "longitude").toPandas()

    # Create GeoDataFrames for the latlong data and countries data
    gdf = gpd.GeoDataFrame(
        latlong_pddf,
        geometry=latlong_pddf.apply(
            lambda row: Point(row["longitude"], row["latitude"]), axis=1
        ),
    )
    countries_gdf = gpd.GeoDataFrame(
        countries,
        geometry=countries.apply(
            lambda row: Point(row["longitude"], row["latitude"]), axis=1
        ),
    )

    # Set the appropriate CRS for the GeoDataFrames.
    gdf.set_crs(epsg=4326, inplace=True)
    countries_gdf.set_crs(epsg=4326, inplace=True)

    # Find the nearest country for each point in the latlong data.
    near_df, near_countries = ckdnearest(gdf, countries_gdf, threshold=30)

    # Merge the nearest country data with the latlong data.
    near_countries.reset_index(inplace=True, names="join_idx")
    near_df.reset_index(inplace=True, names="join_idx")
    result = pd.merge(near_df, near_countries[["join_idx", "iso3"]], how="left")

    # Create a DataFrame with the index and iso3 columns.
    latlong_pddf = result[["index", "iso3"]]
    latlong_df = spark.createDataFrame(latlong_pddf)
    return indexed, latlong_df


def extract_countries_from_cities(
    missing_iso3_df: DataFrame,
    cities_df: DataFrame,
    countries_df: DataFrame,
) -> DataFrame:
    """Extracts the country code (ISO3166) from the city name.

    :param missing_iso3_df: DataFrame consisting of records with missing iso3 values.
    :type missing_iso3_df: DataFrame
    :param cities_df: DataFrame containing the cities data.
    :type cities_df: DataFrame
    :param countries_df: DataFrame containing the countries data.
    :type countries_df: DataFrame
    :return: DataFrame containing the index and iso3 columns for joining.
    :rtype: DataFrame
    """

    # Extract the city name from the tweet_location column.
    missing_iso3_df = (
        missing_iso3_df.withColumn(
            "city",
            F.when(
                F.col("tweet_location").isNotNull(),
                F.split(F.col("tweet_location"), ",")[0],
            ).otherwise(None),
        )
        .withColumn("twitter_username", F.col("name"))
        .drop("name")  # ! Replace the name column with twitter_username
    )

    missing_iso3_df = (
        missing_iso3_df.join(
            other=cities_df.select("name", "country_name"),
            on=(F.col("city") == F.col("name")),
            how="left",
        )
        .drop("name")  # ! Remove the city name column.
        .drop("iso3")  # ! Remove the iso3 column, it should be all null anyway.
        .join(
            other=countries_df.select("iso3", "name"),
            on=(F.col("country_name") == F.col("name")),
            how="left",
        )
        .drop("name")  # ! Remove the country name column.
        .filter(F.col("iso3").isNotNull())
        .select("index", "iso3")
    )
    return missing_iso3_df


def main(
    data_dir: str | os.PathLike | Path = DATA_DIR,
    processed_csv_dir: str | os.PathLike | Path = PROCESSED_CSV_DIR,
    output_dir: str | os.PathLike | Path = OUTPUT_DIR,
):
    """The main function that reads the processed CSV file, extracts the
    latitude and longitude, city name, to determine the country code (ISO3166)
    and writes the output to a CSV file in the output directory.

    :param data_dir: Directory which stores all the data, defaults to DATA_DIR
    :type data_dir: str | os.PathLike | Path, optional
    :param processed_csv_dir: Directory which stores the processed csv from task 1,
        defaults to PROCESSED_CSV_DIR
    :type processed_csv_dir: str | os.PathLike | Path, optional
    :param output_dir: Output directory for the processed dataset ready for MapReduce,
        defaults to OUTPUT_DIR
    :type output_dir: str | os.PathLike | Path, optional
    """

    # Create a Spark session
    spark = (
        SparkSession.builder.appName("Airline Twitter Sentiment Analysis")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

    # Get the processed CSV file path
    processed_csv = get_processed_csv_path(processed_csv_dir)
    if not processed_csv:
        deduplicate_data(data_dir, spark, processed_csv_dir)
        processed_csv = get_processed_csv_path(processed_csv_dir)

    df = spark.read.csv(processed_csv, header=True, inferSchema=True)

    # Load the cities and countries data
    cities_df, countries, countries_df = load_cities_countries_json(data_dir, spark)

    # Extract the countries from the latlong data
    indexed, latlong_df = extract_countries_from_latlong(spark, df, countries)

    # Find those with missing iso3 values.
    indexed = indexed.join(latlong_df, on="index", how="left")
    missing_iso3_df = indexed.filter(F.col("iso3").isNull())

    # Find the missing iso3 values by matching the city name with the countries data.
    extracted_countries_df = extract_countries_from_cities(
        missing_iso3_df, cities_df, countries_df
    )

    # Merge the extracted countries with the latlong data.
    merged_iso3_df = extracted_countries_df.union(latlong_df)
    merged_indexed_df = (
        indexed.drop("iso3")
        .join(merged_iso3_df, on="index", how="left")
        .dropna(subset=["iso3"])
        .drop("index", "tweet_location", "tweet_coord", "latitude", "longitude")
    )

    # Write the output to a CSV file
    merged_indexed_df.repartition(1).write.csv(
        os.path.join(output_dir), header=True, mode="overwrite", quote='"'
    )


def load_cities_countries_json(
    data_dir: str | os.PathLike | Path, spark: SparkSession
) -> tuple[DataFrame, pd.DataFrame, DataFrame]:
    """Loads the cities and countries data from the JSON files.

    :param data_dir: Directory containing the cities and countries JSON files.
    :type data_dir: str | os.PathLike | Path
    :param spark: Spark session object
    :type spark: SparkSession
    :return: Tuple containing the cities DataFrame, countries DataFrame, and countries Spark DataFrame
    :rtype: tuple[DataFrame, pd.DataFrame, DataFrame]
    :raises FileNotFoundError: If the cities.json or countries.json file are not found.
    """
    try:
        with open(os.path.join(data_dir, "cities.json"), "r", encoding="utf-8") as f:
            cities = json.load(f)
            cities = pd.DataFrame.from_records(cities, index="id")
            cities_df = spark.createDataFrame(cities).dropDuplicates(["name"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "cities.json not found, see https://github.com/dr5hn/countries-states-cities-database"
        ) from exc

    try:
        with open(os.path.join(data_dir, "countries.json"), "r", encoding="utf-8") as f:
            countries = json.load(f)
            for country in countries:
                del country["timezones"]
                del country["translations"]
            countries = pd.DataFrame.from_records(countries, index="id")
            countries["longitude"] = cities["longitude"].astype(float)
            countries["latitude"] = cities["latitude"].astype(float)
            countries_df = spark.createDataFrame(countries)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "countries.json not found, see https://github.com/dr5hn/countries-states-cities-database"
        ) from exc

    return cities_df, countries, countries_df


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default=DATA_DIR)
    args.add_argument("--processed_csv_dir", type=str, default=PROCESSED_CSV_DIR)
    args.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    main(**vars(args.parse_args()))
