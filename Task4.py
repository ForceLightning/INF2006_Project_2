import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from load_clean_data import load_clean_data

# Task : To calculate the mean and median values of the trusting points for each channel.


if __name__ == "__main__":
    # Create a SparkSession
    spark = SparkSession.builder.appName("Task4").getOrCreate()

    # Load the data
    df = load_clean_data("./data", spark)
    
    # Group by channel and calculate the mean and median of the trusting points
    df_grouped = df.groupBy("channel").agg(
        mean_trusting = ("trusting", "mean"),
        median_trusting = ("trusting", "median")
    )
    
    # Show the results
    df_grouped.show()
    
