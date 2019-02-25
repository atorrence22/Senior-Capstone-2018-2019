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
    # initiates a SparkContext which is necessary for accessing data in Spark
    sc = SparkContext()
    sqlContext = SQLContext(sc)
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

    AFBF = AFBF.withColumn('POOLID', AFBF.POOLID.cast(IntegerType()))
    SD_CONTAINERS = SD_CONTAINERS.withColumn('POOLID', SD_CONTAINERS.POOLID.cast(IntegerType()))
    SD_CHUNK_LOCATIONS = SD_CHUNK_LOCATIONS.withColumn('CHUNKID', SD_CHUNK_LOCATIONS.CHUNKID.cast(LongType()))
    SD_CHUNK_LOCATIONS = SD_CHUNK_LOCATIONS.withColumn('POOLID', SD_CHUNK_LOCATIONS.POOLID.cast(IntegerType()))

    tape = [row['POOLID'] for row in AFBF.select("POOLID").distinct().collect()]
    cloud = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('3|4')).select('POOLID').distinct().collect()])
    directory = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('1|2')).select('POOLID').distinct().collect()]) - cloud

    merge = BACKUP_OBJECTS.join(AFBF, BACKUP_OBJECTS['OBJID'] == AFBF['BFID'], how='left')
    merge = merge.join(SDRO, ['OBJID'], how='left')
    merge = merge.withColumn('POOLID', merge.POOLID.cast(IntegerType()))
    merge = merge.withColumn('CHUNKID', merge.CHUNKID.cast(LongType()))

    cloud_chunkid = []

    for poolid in cloud:
        rows = SD_CHUNK_LOCATIONS.select(SD_CHUNK_LOCATIONS.CHUNKID).filter(F.when(SD_CHUNK_LOCATIONS.POOLID == poolid, True).otherwise(False)).distinct().collect()
        cloud_chunkid.extend([row['CHUNKID'] for row in rows])

    directory_chunkid = []

    for poolid in directory:
        rows = SD_CHUNK_LOCATIONS.select(SD_CHUNK_LOCATIONS.CHUNKID).filter(F.when(SD_CHUNK_LOCATIONS.POOLID == poolid, True).otherwise(False)).distinct().collect()
        directory_chunkid.extend([row['CHUNKID'] for row in rows])

    def assign_output(poolid, chunkid):
        if poolid:
            return 0
        if chunkid in directory_chunkid:
            return 1
        elif chunkid:
            return 2
        else:
            return None

    output = udf(assign_output, IntegerType())

    merge = merge.withColumn('OUTPUT', output(merge.POOLID, merge.CHUNKID))
    merge = merge.filter(merge.OUTPUT.isNotNull())
    return merge


def extract_features(feature_list):
    merged_data = combine()
    feature_list.append('OUTPUT')
    merged_data = merged_data.withColumn("BFSIZE", merged_data["BFSIZE"].cast(LongType()))
    merged_data = merged_data.withColumn("HDRSIZE", merged_data["HDRSIZE"].cast(LongType()))
    merged_data = merged_data.withColumn("OUTPUT", merged_data["OUTPUT"].cast(IntegerType()))
    merged_data = merged_data.filter(merged_data.OUTPUT.isNotNull())
    merged_data = merged_data.filter(merged_data.BFSIZE.isNotNull())
    merged_data = merged_data.filter(merged_data.HDRSIZE.isNotNull())
    merged_data = merged_data.select(feature_list)
    merged_data = merged_data.repartition(400)
    merged_data.write.csv(output_dir + "/feature_extraction")
    sc.stop()

def main():
    extract_features(['BFSIZE', 'HDRSIZE'])
    return 0

if __name__ == "__main__":
    main()
