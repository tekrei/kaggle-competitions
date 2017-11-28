import time

import pandas as pd
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.appName("Spark for PSSDP").getOrCreate()

spark.sparkContext.setCheckpointDir('checkpoint/')

train_data = spark.read.csv(
    "../input/train.csv", header="true", inferSchema="true")


train_data = train_data.na.fill(-1)

print("loaded training data")

ignore = ['id', 'target']
assembler = VectorAssembler(
    inputCols=[x for x in train_data.columns if x not in ignore],
    outputCol='features')

train_data = (assembler.transform(train_data).select("target", "features"))

print("assembled features")

(train_data, valid_data) = train_data.randomSplit([0.75, 0.25])

iteration = 1000
gbt = GBTClassifier(labelCol="target",
                    featuresCol="features", maxIter=iteration)

evaluator = BinaryClassificationEvaluator(labelCol="target")

model = gbt.fit(train_data)

print("trained GBT classifier:%s" % model)

# Select (prediction, true label) and compute test error
auc_roc = evaluator.evaluate(model.transform(valid_data))
print("AUC ROC = %g" % auc_roc)
gini = (1 - 2 * auc_roc)
print("GINI ~=%g" % gini)

# prepare submission
test_data = spark.read.csv(
    "../input/test.csv", header="true", inferSchema="true")
test_data = test_data.na.fill(-1)
print("loaded testing data")

predictions = model.transform(
    assembler.transform(test_data).select("features"))
print("predicted testing data")

# extract ids from test data
ids = test_data.select("id").rdd.map(lambda x: int(x[0]))

# we should provide probability of 2nd class
targets = predictions.select("probability").rdd.map(lambda x: float(x[0][1]))

# create data frame consists of id and probabilities
submission = spark.createDataFrame(ids.zip(targets), StructType([StructField(
    "id", IntegerType(), True), StructField("target", FloatType(), True)]))

# store results after coalescing
submission.coalesce(1).write.csv('%d-%g-%s.csv' %
                                 (iteration, gini, time.time()), header="true")
print("exported predictions for submission")
spark.stop()
