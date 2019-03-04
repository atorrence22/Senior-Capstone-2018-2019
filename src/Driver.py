import Process
import Learn

from pyspark import SparkContext
from pyspark.sql import SparkSession

import click

@click.command()
@click.argument('features')
@click.argument('algorithm', type=click.Choice(['RandomForest', 'GradientBoosting', 'MultinomialRegression', 'LinearSVC']))
@click.option('--seed', '-s', default=None, type=int)
@click.option('--maxIter',  default=20)
@click.option('--stepSize', default=0.1)
@click.option('--numTrees', default=20)
@click.option('--maxDepth', default=5)
@click.option('--regParam', default=0.0)
@click.option('--elasticNetParam', default=0.0)
@click.option('--threshold', default=0.5)
def main(features, algorithm, seed, maxiter, stepsize, numtrees, maxdepth, regparam, elasticnetparam, threshold):
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    feature_list = features.split()
    if algorithm == 'RandomForest':
        # python Driver.py 'BFSIZE HDRSIZE NODETYPE NODESTATE METADATASIZE STG_HINT' RandomForest
        Learn.randomForest(feature_list, maxDepth=maxdepth, numTrees = 20, seed=seed)
    elif algorithm == 'GradientBoosting':
        Learn.gradientBoosting(feature_list, maxIter=maxiter, stepSize=stepsize)
    elif algorithm == 'MultinomialRegression':
        # python Driver.py 'BFSIZE HDRSIZE NODETYPE NODESTATE METADATASIZE' MultinomialRegression --maxIter 10 --regParam 0.3 --elasticNetParam 0.8 --threshold 0.5
        # without metadatasize
        Learn.multinomialRegression(feature_list, maxIter=maxiter, regParam=regparam, elasticNetParam = elasticnetparam, threshold=threshold)
    elif algorithm == 'LinearSVC':
        Learn.linearSVC(feature_list, maxIter=maxiter, regParam=regparam, threshold=threshold)

    sc.stop()
if __name__ == "__main__":
    main()
