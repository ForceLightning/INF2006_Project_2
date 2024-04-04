import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, when, regexp_replace, count, desc, collect_list, coalesce, lit
from pyspark.sql.window import Window
from util import load_data

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Airline Twitter Sentiment Analysis") \
    .getOrCreate()

df = load_data()

# Removing duplicates based on the tweet and id
df_deduplicated = df.dropDuplicates(['text', '_id'])

# Remove \n and \r \t from the text column
df_deduplicated = df_deduplicated.withColumn("text", regexp_replace("text", "\n|\r|\t", ""))

# Replace all double quotes with single quotes
df_deduplicated = df_deduplicated.withColumn("text", regexp_replace("text", "\"", "'"))

# Filling missing values in the '_missed' column with False
df_deduplicated = df_deduplicated.withColumn("_missed", when(df_deduplicated["_missed"].isNull(), False).otherwise(df_deduplicated["_missed"].cast("boolean")))

# Filling missing values in the 'retweet_count' column with 0
df_deduplicated = df_deduplicated.withColumn("retweet_count", when(df_deduplicated["retweet_count"].isNull(), 0).otherwise(df_deduplicated["retweet_count"]))

# Update 'airline_sentiment' column with values from 'airline_sentiment_gold' where 'airline_sentiment_gold' is not null
df_deduplicated = df_deduplicated.withColumn("airline_sentiment", when(df_deduplicated["airline_sentiment_gold"].isNotNull(), df_deduplicated["airline_sentiment_gold"]).otherwise(df_deduplicated["airline_sentiment"]))

# Split the 'negativereason_gold' column by '&' and create new columns
df_deduplicated = df_deduplicated.withColumn("negativereason_split", split("negativereason_gold", "\n"))

# Extract values from the split column into separate columns
df_deduplicated = df_deduplicated.withColumn("negativereason1", df_deduplicated.negativereason_split.getItem(0)) \
                                 .withColumn("negativereason2", df_deduplicated.negativereason_split.getItem(1))

# If negativereason1 is null, use the value in negativereason
df_deduplicated = df_deduplicated.withColumn("negativereason1", when(df_deduplicated["negativereason1"].isNull(), df_deduplicated["negativereason"]).otherwise(df_deduplicated["negativereason1"]))

# Drop the intermediate split column
df_deduplicated = df_deduplicated.drop("negativereason_split")

# Filling missing values in 'negativereason1' and 'negativereason2' with 'Unknown'
df_deduplicated = df_deduplicated.withColumn("negativereason1", when(df_deduplicated["negativereason1"].isNull(), "Unknown").otherwise(df_deduplicated["negativereason1"]))
df_deduplicated = df_deduplicated.withColumn("negativereason2", when(df_deduplicated["negativereason2"].isNull(), "Unknown").otherwise(df_deduplicated["negativereason2"]))

# Group by country and user_timezone, count occurrences, and order by count in descending order
grouped_df = df_deduplicated.groupBy('_country', 'user_timezone') \
                            .agg(count('*').alias('count')) \
                            .orderBy('_country', desc('count'))

# Get the most common user_timezone for each country
most_common_user_timezone = grouped_df.groupBy('_country') \
                                    .agg(collect_list('user_timezone').getItem(0).alias('most_common_user_timezone'))

# Left join to fill missing values in 'user_timezone' with the most common value for the corresponding country
df_deduplicated = df_deduplicated.join(most_common_user_timezone, on='_country', how='left') \
                                 .withColumn('user_timezone', coalesce(col('user_timezone'), col('most_common_user_timezone'), lit('Unknown')))

# Drop the intermediate 'grouped_df' DataFrame
grouped_df.unpersist()

                                 
# Fill missing values in 'user_timezone' with 'Unknown'
df_deduplicated = df_deduplicated.withColumn('user_timezone', when(col('user_timezone').isNull(), 'Unknown').otherwise(col('user_timezone')))

# Drop inconsistent data columns and columns that have been split
# _tainted is dropped because all the values are False after deduplication
df_deduplicated = df_deduplicated.drop("airline_sentiment_gold", "negativereason_gold", "negativereason", "_region", "_city", "tweet_location", "tweet_coord", "_country", "_tainted", "most_common_user_timezone")

# Show the resulting DataFrame
# df_deduplicated.show()

# Write the DataFrame to a CSV file
# df_deduplicated.write.csv('./Twitter_Airline Dataset/processed', header=True, mode='overwrite')

df_deduplicated.repartition(1).write.csv("./data/processed", header=True, mode='overwrite', quote='\"')