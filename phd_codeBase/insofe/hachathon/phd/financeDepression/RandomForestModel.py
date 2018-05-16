from __future__ import print_function
from io import open
import sys
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
from pyspark.sql.functions import col, when, lit, avg
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.appName("FinDep").config("spark.debug.maxToStringFields", 1000).getOrCreate()

# dirVal = "/user/1533B33/B33PHD" + sys.argv[1]
dirVal = "E:\\Insofe_BigData\\PHD\\PART2"
train1 = "Train.csv"
trainAdditional = "Train_AdditionalData.csv"

test1 = "Test.csv"
testAdditional = "Test_AdditionalData.csv"


def f_replace_space_col(actual_df):
    for col_name in actual_df.columns:
        newdf = actual_df.withColumn(col_name, regexp_replace(col_name, " ", "_"))
        return newdf


# Read this data and create a data frame and verify the dataframe.
train1DF = spark.read.csv(
    dirVal + '\\' + train1,
    header=True, inferSchema=True) \
    .cache()

f_replace_space_col(train1DF)

train1DF.createOrReplaceTempView("train1DF")

trainAdditionalDF = spark.read.csv(
    dirVal + '\\' + trainAdditional,
    header=True, inferSchema=True) \
    .cache()

trainAdditionalDF.createOrReplaceTempView("trainAdditionalDF")

test1DF = spark.read.csv(
    dirVal + '\\' + test1,
    header=True, inferSchema=True) \
    .cache()

test1DF.createOrReplaceTempView("test1DF")

testAdditionalDF = spark.read.csv(
    dirVal + '\\' + testAdditional,
    header=True, inferSchema=True) \
    .cache()

testAdditionalDF.createOrReplaceTempView("testAdditionalDF")

print("count train1DF => ", train1DF.count())
print("count trainAdditionalDF => ", trainAdditionalDF.count())
print("count test1DF => ", test1DF.count())
print("count testAdditionalDF => ", testAdditionalDF.count())

avgNoc = str(train1DF.agg(avg(col("Number_of_Cylinders"))).first()[0])

v_sql = " select DISTINCT " \
        "case when trim(y) = 'pass' then 1 else 0 end as y," \
        "case when trim(Number_of_Cylinders) is NULL then " + (avgNoc) + " else trim(Number_of_Cylinders) end as Number_of_Cylinders," \
        "trim(material_grade) as material_grade," \
        "trim(Lubrication) as Lubrication," \
        "trim(Valve_Type) as Valve_Type," \
        "trim(Bearing_Vendor) as Bearing_Vendor," \
        "trim(Fuel_Type) as Fuel_Type," \
        "trim(Compression_ratio) as Compression_ratio," \
        "trim(cam_arrangement) as cam_arrangement," \
        "trim(Cylinder_arragement) as Cylinder_arragement," \
        "trim(Turbocharger) as Turbocharger," \
        "trim(Varaible_Valve_Timing_VVT) as Varaible_Valve_Timing_VVT," \
        "trim(Cylinder_deactivation) as Cylinder_deactivation," \
        "trim(Direct_injection) as Direct_injection," \
        "trim(main_bearing_type) as main_bearing_type," \
        "trim(displacement) as displacement," \
        "trim(piston_type) as piston_type," \
        "trim(Max_Torque) as Max_Torque," \
        "trim(Peak_Power) as Peak_Power," \
        "trim(Crankshaft_Design) as Crankshaft_Design," \
        "trim(Liner_Design) as Liner_Design" \
        ", case when TestA is null and TestA is null then 2 " \
        " when TestA is not null and TestA is null then 3  " \
        "when TestA is null and TestA is not null then 3  " \
        "when TestA is not null and TestA is not null then 1 end as rating_tests" \
        " from train1DF a LEFT JOIN trainAdditionalDF b ON (TestA = ID OR TestB = ID)"

newTrainDF = spark.sql(v_sql).persist()

v_sql = " select DISTINCT trim(ID) as ID," \
        "trim(Number_of_Cylinders) as Number_of_Cylinders," \
        "trim(material_grade) as material_grade," \
        "trim(Lubrication) as Lubrication," \
        "trim(Valve_Type) as Valve_Type," \
        "trim(Bearing_Vendor) as Bearing_Vendor," \
        "trim(Fuel_Type) as Fuel_Type," \
        "trim(Compression_ratio) as Compression_ratio," \
        "trim(cam_arrangement) as cam_arrangement," \
        "trim(Cylinder_arragement) as Cylinder_arragement," \
        "trim(Turbocharger) as Turbocharger," \
        "trim(Varaible_Valve_Timing_VVT) as Varaible_Valve_Timing_VVT," \
        "trim(Cylinder_deactivation) as Cylinder_deactivation," \
        "trim(Direct_injection) as Direct_injection," \
        "trim(main_bearing_type) as main_bearing_type," \
        "trim(displacement) as displacement," \
        "trim(piston_type) as piston_type," \
        "trim(Max_Torque) as Max_Torque," \
        "trim(Peak_Power) as Peak_Power," \
        "trim(Crankshaft_Design) as Crankshaft_Design," \
        "trim(Liner_Design) as Liner_Design" \
        ", case when TestA is null and TestA is null then 2 " \
        " when TestA is not null and TestA is null then 3  " \
        "when TestA is null and TestA is not null then 3  " \
        "when TestA is not null and TestA is not null then 1 end as rating_tests" \
        " from train1DF a LEFT JOIN trainAdditionalDF b ON (TestA = ID OR TestB = ID)"

newTestDF = spark.sql(v_sql).persist()

testNARemoved = newTestDF.na.fill("999999").withColumn("rating_tests", col('rating_tests').cast("Decimal")) \
    .withColumn("Number_of_Cylinders", col('Number_of_Cylinders').cast("Decimal")) \
    .select("ID",
            "Number_of_Cylinders",
            "material_grade",
            "Lubrication",
            "Valve_Type",
            "Bearing_Vendor",
            "Fuel_Type",
            "Compression_ratio",
            "cam_arrangement",
            "Cylinder_arragement",
            "Turbocharger",
            "Varaible_Valve_Timing_VVT",
            "Cylinder_deactivation",
            "Direct_injection",
            "main_bearing_type",
            "displacement",
            "piston_type",
            "Max_Torque",
            "Peak_Power",
            "Crankshaft_Design",
            "Liner_Design",
            "rating_tests") \
    .na.fill(
    {'ID': '-1', 'Number_of_Cylinders': '-1', 'material_grade': '-1', 'Lubrication': '-1', 'Valve_Type': '-1',
     'Bearing_Vendor': '-1', 'Fuel_Type': '-1', 'Compression_ratio': '-1', 'cam_arrangement': '-1',
     'Cylinder_arragement': '-1', 'Turbocharger': '-1', 'Varaible_Valve_Timing_VVT': '-1',
     'Cylinder_deactivation': '-1', 'Direct_injection': '-1', 'main_bearing_type': '-1', 'displacement': '-1',
     'piston_type': '-1', 'Max_Torque': '-1', 'Peak_Power': '-1', 'Crankshaft_Design': '-1', 'Liner_Design': '-1',
     'rating_tests': '-1'})

data_df = newTrainDF.withColumn("rating_tests", col('rating_tests').cast("Decimal")) \
    .withColumn("Number_of_Cylinders", col('Number_of_Cylinders').cast("Decimal")) \
    .select("y",
            "Number_of_Cylinders",
            "material_grade",
            "Lubrication",
            "Valve_Type",
            "Bearing_Vendor",
            "Fuel_Type",
            "Compression_ratio",
            "cam_arrangement",
            "Cylinder_arragement",
            "Turbocharger",
            "Varaible_Valve_Timing_VVT",
            "Cylinder_deactivation",
            "Direct_injection",
            "main_bearing_type",
            "displacement",
            "piston_type",
            "Max_Torque",
            "Peak_Power",
            "Crankshaft_Design",
            "Liner_Design",
            "rating_tests")

colNames = data_df.columns
for x in colNames:
    data_df = data_df \
        .withColumn(x, when(col(x).isin(['', 'NA'] or col(x).isNull ), lit("0")).otherwise(data_df[x]))

# data_df.show(100)
data_df.fillna("-1")
# data_df.coalesce(1).write.option("header", True).csv("E:\\Insofe_BigData\\PHD\\PART2\\data_df.csv")
# exit(1)


def get_dummy(df, categoricalCols, continuousCols, labelCol):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import col

    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                for c in categoricalCols]

    # default setting: dropLast=True
    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                              outputCol="{0}_encoded".format(indexer.getOutputCol()))
                for indexer in indexers]

    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                          + continuousCols, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    model = pipeline.fit(df)
    data = model.transform(df)

    data = data.withColumn('label', col(labelCol))

    return data.select('features', 'label')


# COMMAND ----------

# Iterate over all the variables in the categorical variables list and run the function that is defined in the previous step
# on each of the variable - This function will return new columns appended to the existing dataframe.
# 'Number_of_Cylinders',
num_cols = [ 'rating_tests']
catcols = ['material_grade', 'Lubrication', 'Valve_Type', 'Bearing_Vendor', 'Fuel_Type', 'Compression_ratio',
           'cam_arrangement', 'Cylinder_arragement', 'Turbocharger', 'Varaible_Valve_Timing_VVT',
           'Cylinder_deactivation', 'Direct_injection',
           'main_bearing_type', 'displacement', 'piston_type', 'Max_Torque', 'Peak_Power', 'Crankshaft_Design',
           'Liner_Design']

labelCol = 'y'

data = get_dummy(data_df, catcols, num_cols, labelCol)
data.show(5)
# Index labels, adding metadata to the label column
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol='label',
                             outputCol='indexedLabel').fit(data)
labelIndexer.transform(data).show(5, True)

from pyspark.ml.feature import VectorIndexer

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", \
                               outputCol="indexedFeatures", \
                               maxCategories=3).fit(data)
featureIndexer.transform(data).show(5, True)

(trainingData, testData) = data.randomSplit([0.7, 0.3])

trainingData.show(5, False)
testData.show(5, False)

from pyspark.ml.classification import LogisticRegression

logr = LogisticRegression(featuresCol='indexedFeatures', labelCol='indexedLabel')

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, logr, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)
# Select example rows to display.
predictions.select("features", "label", "predictedLabel").show(5)

# predictions.coalesce(1).write.option("header",True).mode(sav).csv(dirVal + '/' + "predictions.csv")

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

lrModel = model.stages[2]
trainingSummary = lrModel.summary

# Obtain the objective per iteration
# objectiveHistory = trainingSummary.objectiveHistory
# print("objectiveHistory:")
# for objective in objectiveHistory:
#     print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
trainingSummary.roc.show(5)
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# Set the model threshold to maximize F-Measure
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head(5)
