from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import FloatType

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io

# change to be reflective of your environment
data_dir = '/home/cole/Workspace/School/Capstone/data/first_data_set/TestData/'

# change to match your environment
output_dir = data_dir + "/merge_data"

def linearSVC(df, feature_list=['BFSIZE', 'HDRSIZE', 'NODETYPE'], maxIter=100, regParam=0.0, threshold=0.0):
    # Checks if there is a SparkContext running if so grab that if not start a new one
    # sc = SparkContext.getOrCreate()
    # sqlContext = SQLContext(sc)
    # sqlContext.setLogLevel('INFO')

    vector_assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    df_temp = vector_assembler.transform(df)

    df = df_temp.select(['label', 'features']).withColumnRenamed('label', 'label')

    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    lsvc = LinearSVC(maxIter=10, regParam=0.1)
    model = lsvc.fit(trainingData)

    predictions = model.transform(testData)
    # predictions.select("prediction", "label").show(40)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(predictions)

    # test distribution of outputs
    total = df.select('label').count()
    disk = df.filter(df.label == 0).count()
    cloud = df.filter(df.label == 1).count()

    # print outputs
    print('LinearSVC')
    print(feature_list)
    print(' Cloud %{}'.format((cloud/total) * 100))
    print(' Disk %{}'.format((disk/total) * 100))

    # print(" Test Error = {}".format((1.0 - accuracy) * 100))
    # print(" Test Accuracy = {}\n".format(accuracy * 100))
    print(" Test AUC = {}\n".format(auc * 100))

    misses = predictions.filter(predictions.label != predictions.prediction)
    # now get percentage of error
    disk_misses = misses.filter(misses.label == 0).count()
    cloud_misses = misses.filter(misses.label == 1).count()

    print(' Cloud Misses %{}'.format((cloud_misses/cloud) * 100))
    print(' Disk Misses %{}'.format((disk_misses/disk) * 100))

    return auc, 'LinearSVC: {}'.format(auc), model


def multinomialRegression(df, feature_list=['BFSIZE', 'HDRSIZE', 'NODETYPE'], maxIter = 100, regParam = 0.0, elasticNetParam = 0.0, threshold = 0.5):
    # Checks if there is a SparkContext running if so grab that if not start a new one
    # sc = SparkContext.getOrCreate()
    # sqlContext = SQLContext(sc)
    # sqlContext.setLogLevel('INFO')

    vector_assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    df_temp = vector_assembler.transform(df)

    df = df_temp.select(['label', 'features'])

    (trainingData, testData) = df.randomSplit([0.7, 0.3])
    lr = LogisticRegression(labelCol="label", maxIter= maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
    model = lr.fit(trainingData)
    predictions = model.transform(testData)
    # predictions.select("prediction", "label").show(100)
    # df.select('label').distinct().show()
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    # test distribution of outputs
    total = df.select('label').count()
    tape = df.filter(df.label == 0).count()
    disk = df.filter(df.label == 1).count()
    cloud = df.filter(df.label == 2).count()
    # print outputs
    print('Multinomial Regression')
    print(feature_list)
    print(' Cloud %{}'.format((cloud/total) * 100))
    print(' Disk %{}'.format((disk/total) * 100))
    print(' Tape %{}\n'.format((tape/total) * 100))

    print(" Test Error = {}".format((1.0 - accuracy) * 100))
    print(" Test Accuracy = {}\n".format(accuracy * 100))

    misses = predictions.filter(predictions.label != predictions.prediction)
    # now get percentage of error
    tape_misses = misses.filter(misses.label == 0).count()
    disk_misses = misses.filter(misses.label == 1).count()
    cloud_misses = misses.filter(misses.label == 2).count()

    print(' Cloud Misses %{}'.format((cloud_misses/cloud) * 100))
    print(' Disk Misses %{}'.format((disk_misses/disk) * 100))
    print(' Tape Misses %{}'.format((tape_misses/tape) * 100))

    return accuracy, 'Multinomial Regression: {}'.format(accuracy), model


def randomForest(df, feature_list=['BFSIZE', 'HDRSIZE', 'NODETYPE'], maxDepth = 5, numTrees = 20, seed=None):
    # Checks if there is a SparkContext running if so grab that if not start a new one
    # sc = SparkContext.getOrCreate()
    # sqlContext = SQLContext(sc)
    # sqlContext.setLogLevel('INFO')
    print('Sanity check that Im not doing something dumb like using the label in the feature_list: {}'.format(feature_list))
    vector_assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    df_temp = vector_assembler.transform(df)

    df = df_temp.select(['label', 'features'])

    (trainingData, testData) = df.randomSplit([0.7, 0.3])
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees= numTrees, maxDepth = maxDepth, seed = seed)

    model = rf.fit(trainingData)
    predictions = model.transform(testData)
    # predictions.select("prediction", "label").show(100)
    # df.select('label').distinct().show()
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    # f1|weightedPrecision|weightedRecall|accuracy
    evaluator.setMetricName('accuracy')
    accuracy = evaluator.evaluate(predictions)
    evaluator.setMetricName('f1')
    f1 = evaluator.evaluate(predictions)
    evaluator.setMetricName('weightedPrecision')
    weightedPrecision = evaluator.evaluate(predictions)
    evaluator.setMetricName('weightedRecall')
    weightedRecall = evaluator.evaluate(predictions)

    print('accuracy {}'.format(accuracy))
    print('f1 {}'.format(f1))
    print('weightedPrecision {}'.format(weightedPrecision))
    print('weightedRecall {}'.format(weightedRecall))

    # test distribution of outputs
    total = df.select('label').count()
    tape = df.filter(df.label == 0).count()
    disk = df.filter(df.label == 1).count()
    cloud = df.filter(df.label == 2).count()
    # print outputs
    print('Random Forests')
    print(feature_list)
    print('Total Observations {}'.format(total))
    print(' Cloud %{}'.format((cloud/total) * 100))
    print(' Disk %{}'.format((disk/total) * 100))
    print(' Tape %{}\n'.format((tape/total) * 100))

    print(" Test Error = {}".format((1.0 - accuracy) * 100))
    print(" Test Accuracy = {}\n".format(accuracy * 100))

    misses = predictions.filter(predictions.label != predictions.prediction)
    # now get percentage of error
    tape_misses = misses.filter(misses.label == 0).count()
    disk_misses = misses.filter(misses.label == 1).count()
    cloud_misses = misses.filter(misses.label == 2).count()

    print(' Cloud Misses %{}'.format((cloud_misses/cloud) * 100))
    print(' Disk Misses %{}'.format((disk_misses/disk) * 100))
    print(' Tape Misses %{}'.format((tape_misses/tape) * 100))

    # plt.xlabel("FPR", fontsize=14)
    # plt.ylabel("TPR", fontsize=14)
    # plt.title("ROC Curve", fontsize=14)
    # plt.plot(fp[0:250], tp, linewidth=2)
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = tf.image.decode_png(buf.getvalue(), channels=4)
    # image = tf.expand_dims(image, 0)
    # summary_op = tf.summary.image("ROC Curve", image)
    return accuracy, 'Random Forests: {}'.format(accuracy), model


def gradientBoosting(df, feature_list=['BFSIZE', 'HDRSIZE', 'NODETYPE'], maxIter=20, stepSize=0.1):
    # Checks if there is a SparkContext running if so grab that if not start a new one
    # sc = SparkContext.getOrCreate()
    # sqlContext = SQLContext(sc)
    # sqlContext.setLogLevel('INFO')

    vector_assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    df_temp = vector_assembler.transform(df)

    df = df_temp.select(['label', 'features']).withColumnRenamed('label', 'label')

    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
    model = gbt.fit(trainingData)

    predictions = model.transform(testData)
    #predictions.select("prediction", "label").show(40)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    # accuracy = evaluator.evaluate(predictions, {evaluator.metricName:"Precision"})
    auc = evaluator.evaluate(predictions)

    # test distribution of outputs
    total = df.select('label').count()
    disk = df.filter(df.label == 0).count()
    cloud = df.filter(df.label == 1).count()

    # print outputs
    print('Gradient-Boosted Tree')
    print(' Cloud %{}'.format((cloud/total) * 100))
    print(' Disk %{}'.format((disk/total) * 100))
    print(feature_list)

    # print(" Test Error = {}".format((1.0 - accuracy) * 100))
    # print(" Test Accuracy = {}\n".format(accuracy * 100))
    print(" Test AUC = {}\n".format(auc * 100))

    misses = predictions.filter(predictions.label != predictions.prediction)
    # now get percentage of error
    disk_misses = misses.filter(misses.label == 0).count()
    cloud_misses = misses.filter(misses.label == 1).count()

    print(' Cloud Misses %{}'.format((cloud_misses/cloud) * 100))
    print(' Disk Misses %{}'.format((disk_misses/disk) * 100))

    return auc, 'Gradient Boosted: {}'.format(auc), model


def compare_algorithms():
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    binary = [gradientBoosting, linearSVC]
    multiclass = [randomForest, multinomialRegression]
    binary_results = []
    class_results = []
    print('Binary Function Comparisons')
    for f in binary:
        res = f()
        print(res[1])
        binary_results.append(res)
    print('Multiclass Function Comparisons')
    for f in multiclass:
        res = f()
        print(res[1])
        class_results.append(res)

    binary_results.sort(reverse=True)
    class_results.sort(reverse=True)

    print('Binary Results in order:')
    for res, res_string in binary_results:
        print(res_string)

    print('Multiclass Results in order:')
    for res, res_string in class_results:
        print(res_string)

    sc.stop()

    return 0


def main():
    from Process import extract_features
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    sqlContext = SQLContext(sc)

    feature_list = 'BFSIZE HDRSIZE NODETYPE NODESTATE METADATASIZE'.split()
    merged_data, merged_data_binary = extract_features(feature_list, binary = True, multiclass = True, overwrite = True)
    # merged_data = merged_data.limit(2000)
    results = []

    print('Start Random Forest')
    results.append(randomForest(merged_data, feature_list = feature_list, maxDepth = 5, numTrees = 20, seed=None))
    # print('Start GradientBoosting')
    # results.append(gradientBoosting(merged_data_binary, feature_list = feature_list, maxIter=20, stepSize=0.1))

    print('Results:')
    for result in results:
        print(result)

    sc.stop()
    return 0


if __name__ == "__main__":
    main()
