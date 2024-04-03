import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, when
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Read CSV into PySpark DataFrame") \
    .getOrCreate()

# Read the CSV file into a PySpark DataFrame
df = spark.read.csv('./Twitter_Airline Dataset/combined.csv', header=True, inferSchema=True)

# Removing duplicates based on the tweet and id
df_deduplicated = df.dropDuplicates(['text', '_id'])

# Update 'airline_sentiment' column with values from 'airline_sentiment_gold' where 'airline_sentiment_gold' is not null
df_deduplicated = df_deduplicated.withColumn("airline_sentiment", when(df_deduplicated["airline_sentiment_gold"].isNotNull(), df_deduplicated["airline_sentiment_gold"]).otherwise(df_deduplicated["airline_sentiment"]))

# Split the 'negativereason_gold' column by '&' and create new columns
df_deduplicated = df_deduplicated.withColumn("negativereason_split", split("negativereason_gold", "&"))

# Extract values from the split column into separate columns
df_deduplicated = df_deduplicated.withColumn("negativereason1", df_deduplicated.negativereason_split.getItem(0)) \
                                 .withColumn("negativereason2", df_deduplicated.negativereason_split.getItem(1))

# If negativereason1 is null, use the value in negativereason
df_deduplicated = df_deduplicated.withColumn("negativereason1", when(df_deduplicated["negativereason1"].isNull(), df_deduplicated["negativereason"]).otherwise(df_deduplicated["negativereason1"]))

# Drop the intermediate split column
df_deduplicated = df_deduplicated.drop("negativereason_split")

# Drop inconsistent data columns and columns that have been split
df_deduplicated = df_deduplicated.drop("airline_sentiment_gold", "negativereason_gold", "negativereason", "_region", "_city", "tweet_location", "tweet_coord")

# Show the resulting DataFrame
# df_deduplicated.show()

# Write the DataFrame to a CSV file
df_deduplicated.write.csv('./Twitter_Airline Dataset/processed.csv', header=True, mode='overwrite')

