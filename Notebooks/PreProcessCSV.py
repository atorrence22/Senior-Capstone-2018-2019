from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType
import sys

# initiates a SparkContext which is necessary for accessing data in Spark
sc = SparkContext()
sqlContext = SQLContext(sc)
# change to match your environment
output_dir = "Data/merge_data"
feature_list = sys.argv

# commented code below should be handled in a previous stage
# chosen_csvs = ["Data/SS_POOLS.csv", "Data/SD_CHUNK_LOCATIONS.csv", "Data/ARCHIVE_OBJECTS.csv"]

df = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load(chosen_csvs)

# count starts at 1 because firsy item in passed in list is the name of file
count = 1
for item in feature_list:
    df.withColumn(feature_list[count], df[feature_list[count]].cast("int"))
    count += 1

df.select(feature_list).write.options(header='true').format('com.databricks.spark.csv').save("Data/merge_data/4_features")

test_df = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load("Data/merge_data/4_features/*.csv")
test_df.columns

sc.stop()
