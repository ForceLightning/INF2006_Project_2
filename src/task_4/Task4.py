import findspark

findspark.init()

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import sys

DIR = "src/"
sys.path.append(DIR)
from utils.load_clean_data import load_clean_data

# Task 4 : To calculate the mean and median values of the trusting points for each channel.
if __name__ == "__main__":
    # create a SparkSession
    spark = SparkSession.builder.appName("Task4").getOrCreate()

    # load the data
    df = load_clean_data("./data", spark)

    # group by channel and calculate the mean and median of the trusting points
    df_grouped = df.groupBy("_channel").agg(
        F.mean("_trust").alias("mean_trusting_points"),
        F.median("_trust").alias("median_trusting_points"),
    )

    # DEBUG Show the results
    # print(df_grouped.head(10))

    # output results to a csv file
    df_grouped.repartition(1).write.csv(
        DIR + "task_4/output", header=True, mode="overwrite", quote='"'
    )

    # stop the spark session
    spark.stop()
