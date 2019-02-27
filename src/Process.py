from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import LongType

# change to be reflective of your environment
data_dir = '/home/cole/Workspace/School/Capstone/data/first_data_set/TestData/'

# change to match your environment
output_dir = data_dir + "/merge_data"

def combine():
    # Checks if there is a SparkContext running if so grab that if not start a new one
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)

    # Load all of the CSVs
    AFBF = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir+"/AF_BITFILES.csv"])
    BACKUP_OBJECTS = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir+"/BACKUP_OBJECTS.csv"])
    ARCHIVE_OBJECTS = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir+"/ARCHIVE_OBJECTS.csv"])
    FILESPACES = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir+"/FILESPACES.csv"])
    NODES = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir+"/NODES.csv"])
    SD_CHUNK_COPIES = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir+"/SD_CHUNK_COPIES.csv"])
    SD_CHUNK_LOCATIONS = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir+"/SD_CHUNK_LOCATIONS.csv"])
    SD_CONTAINERS = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir+"/SD_CONTAINERS.csv"])
    SD_NON_DEDUP_LOCATIONS = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir+"/SD_NON_DEDUP_LOCATIONS.csv"])
    SDRO = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir + '/SD_RECON_ORDER.csv'])
    SS_POOLS = sqlContext.read.format('com.databricks.spark.csv').option("header", "true").load([data_dir + '/SS_POOLS.csv'])

    # Recast columns used to get OUTPUT column. We do this because long and int comparisons are faster than strings
    AFBF = AFBF.withColumn('POOLID', AFBF.POOLID.cast(IntegerType()))
    SD_CONTAINERS = SD_CONTAINERS.withColumn('POOLID', SD_CONTAINERS.POOLID.cast(IntegerType()))
    SD_CHUNK_LOCATIONS = SD_CHUNK_LOCATIONS.withColumn('POOLID', SD_CHUNK_LOCATIONS.POOLID.cast(IntegerType()))
    SD_CHUNK_LOCATIONS = SD_CHUNK_LOCATIONS.withColumn('CHUNKID', SD_CHUNK_LOCATIONS.CHUNKID.cast(LongType()))

    # get a set of poolids for each classification
    tape = [row['POOLID'] for row in AFBF.select("POOLID").distinct().collect()]
    # these are sets to remove all occurences that have ANY entry of 1 or 2 instead of ALL entries
    cloud = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('3|4')).select('POOLID').distinct().collect()])
    directory = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('1|2')).select('POOLID').distinct().collect()]) - cloud

    # merge AFBF to BACKUP_OBJECTS, this makes finding the tape objects faster
    merge = BACKUP_OBJECTS.join(AFBF, BACKUP_OBJECTS['OBJID'] == AFBF['BFID'], how='left')
    merge = merge.join(SDRO, ['OBJID'], how='left')
    # recast for performance
    merge = merge.withColumn('POOLID', merge.POOLID.cast(IntegerType()))
    merge = merge.withColumn('CHUNKID', merge.CHUNKID.cast(LongType()))

    # Get dataframes that have entries that have POOLIDs that are in their respective sets
    cloud_chunkid = SD_CHUNK_LOCATIONS.where(SD_CHUNK_LOCATIONS.POOLID.isin(cloud)).select(SD_CHUNK_LOCATIONS.CHUNKID)
    directory_chunkid = SD_CHUNK_LOCATIONS.where(SD_CHUNK_LOCATIONS.POOLID.isin(directory)).select(SD_CHUNK_LOCATIONS.CHUNKID)
    # Find the intersect for each, theres a wierd bug that happens if you skip this step because its a different column object from the merge
    directory_col = merge.select('CHUNKID').intersect(directory_chunkid)
    cloud_col = merge.select('CHUNKID').intersect(cloud_chunkid)

    # construct new dataframe from a copy of the old dataframe with the OUTPUT column augmented on
    merge = merge.withColumn('OUTPUT', F.when(merge.POOLID.isNotNull(), 0).when(merge.CHUNKID.isin(directory_col.CHUNKID), 1).when(merge.CHUNKID.isin(cloud_col.CHUNKID), 2))
    merge = merge.filter(merge.OUTPUT.isNotNull())

    return merge


def extract_features(feature_list):
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    # Here there should be a dictionary with every feature where the value is the cast type

    merged_data = combine()
    # Append OUTPUT to feature list because it is not a feature but necessary
    feature_list.append('OUTPUT')
    # make a copy of the dataframe with only the feature columns and OUTPUT
    merged_data = merged_data.select(feature_list)
    # Right here we need to go through the feature list and properly cast and screen (filter out nulls) each value in the columns
    merged_data.write.csv(output_dir + "/feature_extraction")
    return merged_data


def main():
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    extract_features(['BFSIZE', 'HDRSIZE'])
    sc.stop()
    return 0



if __name__ == "__main__":
    main()
