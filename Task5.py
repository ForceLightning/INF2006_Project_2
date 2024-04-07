from util import load_data
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
import os
from load_clean_data import load_clean_data
from get_sentiment import SentimentAnalysis
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import FloatType, StringType
import pandas as pd

nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def run():
    print("Running Task 2")
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

    spark = SparkSession.builder \
        .appName("Airline Twitter Sentiment Analysis") \
        .getOrCreate()

    df = load_clean_data(
        data_dir='Twitter_Airline Dataset', spark_session=spark)

    sentiment_class = SentimentAnalysis()
    df = df.withColumn('sentiment', udf(
        sentiment_class.get_sentiment, StringType())(col('text')))

    df = df.withColumn('sentiment_without_stop_words', udf(
        sentiment_class.get_sentiment_stop_words_removed, StringType())(col('text')))

    df = df.withColumn('get_sentiment_inverse_if_negative', udf(
        sentiment_class.get_sentiment_inverse_if_negative, StringType())(col('text')))

    df = df.withColumn('get_sentiment_higher_threshold', udf(
        sentiment_class.get_sentiment_higher_threshold, StringType())(col('text')))

    df = df.withColumn('get_sentiment_vader', udf(
        sentiment_class.get_sentiment_vader, StringType())(col('text')))

    to_analyse_columns = ['sentiment', 'sentiment_without_stop_words',
                          'get_sentiment_inverse_if_negative', 'get_sentiment_higher_threshold', 'get_sentiment_vader']

    df = df[df['airline_sentiment_gold'].isNotNull()]

    total_rows = df.count()

    for column in to_analyse_columns:
        is_matching_column = f"{column}_is_matching"
        print(f"Calculating accuracy for {column}")
        df = df.withColumn(is_matching_column, when(
            col(column) == col('airline_sentiment_gold'), 1).otherwise(0))
        matches = df.filter(col(is_matching_column) == 1).count()

        accuracy = (matches / total_rows) * 100

        print(f"Accuracy for {column} is {accuracy}%")

    df.repartition(1).write.csv(
        "./Twitter_Airline Dataset/sentiment", header=True, mode='overwrite', quote='\"')

    print("Task 2 completed")

    spark.stop()


if __name__ == "__main__":
    run()
