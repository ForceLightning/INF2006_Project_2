"""This module performs the preprocessing step (after task 1) for task 3's main
MapReduce job.
"""

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract
from scipy.spatial import cKDTree
from shapely.geometry import Point

DATA_DIR = "data"
PROCESSED_CSV_DIR = os.path.join(DATA_DIR, "processed")
PROCESSED_CSV = [
    os.path.join(PROCESSED_CSV_DIR, file)
    for file in os.listdir(PROCESSED_CSV_DIR)
    if file.endswith(".csv")
]
OUTPUT_DIR = os.path.join(DATA_DIR, "task_3")


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


def main():
    """The main function that reads the processed CSV file, extracts the
    latitude and longitude, city name, to determine the country code (ISO3166)
    and writes the output to a CSV file in the output directory.
    """
    spark = (
        SparkSession.builder.appName("Airline Twitter Sentiment Analysis")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

    df = spark.read.csv(PROCESSED_CSV, header=True, inferSchema=True)

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

    # Extract the latitude and longitude with regex
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

    indexed = df.withColumn("index", F.monotonically_increasing_id())
    latlong_df = indexed.filter(
        F.col("latitude").isNotNull() & F.col("longitude").isNotNull()
    )
    latlong_pddf = latlong_df.select("index", "latitude", "longitude").toPandas()

    with open(os.path.join(DATA_DIR, "cities.json"), "r", encoding="utf-8") as f:
        cities = json.load(f)
        cities = pd.DataFrame.from_records(cities, index="id")
        cities_df = spark.createDataFrame(cities).dropDuplicates(["name"])

    with open(os.path.join(DATA_DIR, "countries.json"), "r", encoding="utf-8") as f:
        countries = json.load(f)
        for country in countries:
            del country["timezones"]
            del country["translations"]
        countries = pd.DataFrame.from_records(countries, index="id")
        countries["longitude"] = cities["longitude"].astype(float)
        countries["latitude"] = cities["latitude"].astype(float)
        countries_df = spark.createDataFrame(countries)

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

    gdf.set_crs(epsg=4326, inplace=True)
    countries_gdf.set_crs(epsg=4326, inplace=True)

    near_df, near_countries = ckdnearest(gdf, countries_gdf, threshold=30)

    near_countries.reset_index(inplace=True, names="join_idx")
    near_df.reset_index(inplace=True, names="join_idx")
    result = pd.merge(near_df, near_countries[["join_idx", "iso3"]], how="left")

    latlong_pddf = result[["index", "iso3"]]
    latlong_df = spark.createDataFrame(latlong_pddf)

    indexed = indexed.join(latlong_df, on="index", how="left")
    missing_iso3_df = indexed.filter(F.col("iso3").isNull())

    missing_iso3_df = (
        missing_iso3_df.withColumn(
            "city",
            F.when(
                F.col("tweet_location").isNotNull(),
                F.split(F.col("tweet_location"), ",")[0],
            ).otherwise(None),
        )
        .withColumn("twitter_username", F.col("name"))
        .drop("name")
    )
    missing_iso3_df = (
        missing_iso3_df.join(
            other=cities_df.select("name", "country_name"),
            on=(F.col("city") == F.col("name")),
            how="left",
        )
        .drop("name")
        .drop("iso3")
        .join(
            other=countries_df.select("iso3", "name"),
            on=(F.col("country_name") == F.col("name")),
            how="left",
        )
        .drop("name")
        .filter(F.col("iso3").isNotNull())
        .select("index", "iso3")
    )
    merged_iso3_df = missing_iso3_df.union(latlong_df)
    merged_indexed_df = (
        indexed.drop("iso3")
        .join(merged_iso3_df, on="index", how="left")
        .dropna(subset=["iso3"])
        .drop("index", "tweet_location", "tweet_coord", "latitude", "longitude")
    )

    merged_indexed_df.repartition(1).write.csv(
        os.path.join(OUTPUT_DIR), header=True, mode="overwrite", quote='"'
    )


if __name__ == "__main__":
    main()
