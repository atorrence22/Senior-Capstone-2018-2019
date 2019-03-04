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

def combine(feature_list, binary=False):
    # Checks if there is a SparkContext running if so grab that if not start a new one
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)

    # TODO: update this to latest standard for reading CSVs
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

    # Recast columns used to get label column. We do this because long and int comparisons are faster than strings
    AFBF = AFBF.withColumn('POOLID', AFBF.POOLID.cast(IntegerType()))
    SD_CONTAINERS = SD_CONTAINERS.withColumn('POOLID', SD_CONTAINERS.POOLID.cast(IntegerType()))
    SD_CHUNK_LOCATIONS = SD_CHUNK_LOCATIONS.withColumn('POOLID', SD_CHUNK_LOCATIONS.POOLID.cast(IntegerType()))
    SD_CHUNK_LOCATIONS = SD_CHUNK_LOCATIONS.withColumn('CHUNKID', SD_CHUNK_LOCATIONS.CHUNKID.cast(LongType()))

    # merge AFBF to BACKUP_OBJECTS, this makes finding the tape objects faster
    AFBF = AFBF.withColumnRenamed('POOLID', 'TAPEPOOLID')

    merge = BACKUP_OBJECTS.join(AFBF, BACKUP_OBJECTS['OBJID'] == AFBF['BFID'], how='left')
    merge = merge.join(SDRO, ['OBJID'], how='left')
    merge = merge.join(SD_CHUNK_LOCATIONS, ['CHUNKID'], how='left')

    # recast for performance
    merge = merge.withColumn('POOLID', merge.POOLID.cast(IntegerType()))
    merge = merge.withColumn('CHUNKID', merge.CHUNKID.cast(LongType()))

    if set(NODES.columns).intersection(set(feature_list)):
        merge = merge.join(NODES, ['NODEID'])


    set5 = False

    if set5:
        # get a set of poolids for each classification
        # these are sets to remove all occurences that have ANY entry of 1 or 2 instead of ALL entries
        cloud3 = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('3')).select('POOLID').distinct().collect()])
        cloud4 = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('4')).select('POOLID').distinct().collect()])
        directory1 = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('1')).select('POOLID').distinct().collect()]) - cloud3 - cloud4
        directory2 = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('2')).select('POOLID').distinct().collect()]) - cloud3 - cloud4

        merge = merge.withColumn('label', F.when(merge.TAPEPOOLID.isNotNull(), 0).when(merge.POOLID.isin(directory1), 1).when(merge.POOLID.isin(directory2), 2).when(merge.POOLID.isin(cloud3), 3).when(merge.POOLID.isin(cloud4), 4))
        merge = merge.filter(merge.label.isNotNull())

        if binary:
            # tape = [row['POOLID'] for row in AFBF.select("POOLID").distinct().collect()]
            cloud = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('3|4')).select('POOLID').distinct().collect()])
            directory = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('1|2')).select('POOLID').distinct().collect()]) - cloud
            bin_merge = merge.withColumn('label', F.when(merge.POOLID.isin(directory), 0).when(merge.POOLID.isin(cloud), 1))
            bin_merge = bin_merge.filter(bin_merge.label.isNotNull())
        else:
            bin_merge = None

    else:
        cloud = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('3|4')).select('POOLID').distinct().collect()])
        directory = set([row['POOLID'] for row in SD_CONTAINERS.filter(SD_CONTAINERS.TYPE.rlike('1|2')).select('POOLID').distinct().collect()]) - cloud

        merge = merge.withColumn('label', F.when(merge.TAPEPOOLID.isNotNull(), 0).when(merge.POOLID.isin(directory), 1).when(merge.POOLID.isin(cloud), 2))
        merge = merge.filter(merge.label.isNotNull())

        if binary:
            bin_merge = merge.withColumn('label', F.when(merge.POOLID.isin(directory), 0).when(merge.POOLID.isin(cloud), 1))
            bin_merge = bin_merge.filter(bin_merge.label.isNotNull())
        else:
            bin_merge = None

    return merge, bin_merge


def extract_features(feature_list, binary = False):
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    # Here there should be a dictionary with every feature where the value is the cast type

    merged_data, merged_data_binary = combine(feature_list, binary)
    feature_list.append('label')

    if binary:
        merged_data_binary = merged_data_binary.select(feature_list)
        merged_data_binary = merged_data_binary.filter(merged_data_binary.BFSIZE.isNotNull())
        merged_data_binary = merged_data_binary.filter(merged_data_binary.HDRSIZE.isNotNull())
        merged_data_binary = merged_data_binary.filter(merged_data_binary.NODETYPE.isNotNull())
        merged_data_binary = merged_data_binary.filter(merged_data_binary.NODESTATE.isNotNull())
        merged_data_binary = merged_data_binary.filter(merged_data_binary.METADATASIZE.isNotNull())
        merged_data_binary = merged_data_binary.filter(merged_data_binary.STG_HINT.isNotNull())
        # merged_data_binary = merged_data_binary.filter(merged_data_binary.FLAGS.isNotNull())

    else:
        merged_data_binary = None

    # Append label to feature list because it is not a feature but necessary
    # make a copy of the dataframe with only the feature columns and label
    merged_data = merged_data.select(feature_list)
    # merged_data.select('label').distinct().show()
    merged_data = merged_data.filter(merged_data.BFSIZE.isNotNull())
    merged_data = merged_data.filter(merged_data.HDRSIZE.isNotNull())
    merged_data = merged_data.filter(merged_data.NODETYPE.isNotNull())
    merged_data = merged_data.filter(merged_data.METADATASIZE.isNotNull())
    merged_data = merged_data.filter(merged_data.NODESTATE.isNotNull())
    merged_data = merged_data.filter(merged_data.STG_HINT.isNotNull())
    # merged_data = merged_data.filter(merged_data.FLAGS.isNotNull())

    merged_data = merged_data.repartition(2000)
    # merged_data.select('label').distinct().show()
    # Right here we need to go through the feature list and properly cast and screen (filter out nulls) each value in the columns
    if binary:
        merged_data_binary = merged_data_binary.repartition(2000)
        merged_data_binary.write.csv(output_dir + "/feature_extraction_binary", header = 'true')

    merged_data.write.csv(output_dir + "/feature_extraction", header = 'true')
    return merged_data, merged_data_binary


def main():
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    feature_list = 'BFSIZE HDRSIZE NODETYPE NODESTATE METADATASIZE STG_HINT'.split()
    extract_features(feature_list, binary=True)
    sc.stop()
    return 0


if __name__ == "__main__":
    main()
