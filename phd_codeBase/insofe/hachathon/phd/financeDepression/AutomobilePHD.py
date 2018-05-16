from __future__ import print_function
from io import open
import sys
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, \
    StandardScaler
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier, GBTClassifier, \
    LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import *
from pyspark.sql.functions import col, when, lit, avg
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("AutoMobile_Model").config("spark.debug.maxToStringFields", 1000).getOrCreate()

# dirVal = "/user/1533B33/B33PHD" + sys.argv[1]
dirVal = "E:\\Insofe_BigData\\PHD\\PART2"
train1 = "Train.csv"
trainAdditional = "Train_AdditionalData.csv"

test1 = "Test.csv"
testAdditional = "Test_AdditionalData.csv"

# Read this data and create a data frame and verify the dataframe.
train1DF = spark.read.csv(
    dirVal + '\\' + train1,
    header=True, inferSchema=True) \
    .cache()


def f_replace_space_col(actual_df):
    for col_name in actual_df.columns:
        newdf = actual_df.withColumn(col_name, regexp_replace(col_name, " ", "_"))
        return newdf


def f_fix_nulls_nas(colNames, df):
    for x in colNames:
        return df.withColumn(x, when(col(x).isin(["", "NA"] or col(x).isNull), lit("0")).otherwise(df[x]))


def call_crossValidator(train_df, test_df, paramGrid, pipeline, evaluator):
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=5,
                              seed=seed)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(train_df).bestModel
    train_predictions_rf = cvModel.transform(train_df)
    test_predictions_rf = cvModel.transform(test_df)

    predictionAndLabels_train = train_predictions_rf.select("prediction", "label")
    predictionAndLabels_test = test_predictions_rf.select("prediction", "label")
    print(train_df.show(10))
    print(train_predictions_rf.show(10))
    train_accuracy = evaluator.evaluate(predictionAndLabels_train)
    test_accuracy = evaluator.evaluate(predictionAndLabels_test)

    print("Train set accuracy = " + str(train_accuracy))
    print("Train Error = %g" % (1.0 - train_accuracy))

    print("Test set accuracy of RF = " + str(test_accuracy))
    print("Test Error = %g" % (1.0 - test_accuracy))
    print('Accuracy value on Test data is', test_accuracy)
    return cvModel


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

# "trim(Varaible_Valve_Timing_VVT) as Varaible_Valve_Timing_VVT,"
# "trim(main_bearing_type) as main_bearing_type,"
v_sql = " select DISTINCT " \
        "case when trim(y) = 'pass' then 1 else 0 end as label," \
        "case when trim(Number_of_Cylinders) is NULL or trim(Number_of_Cylinders) = 'NULL' or trim(Number_of_Cylinders) = ' ' or trim(Number_of_Cylinders) = 'NA' then " + (
            avgNoc) + " else trim(Number_of_Cylinders) end as Number_of_Cylinders," \
                      "trim(material_grade) as material_grade," \
                      "trim(Lubrication) as Lubrication," \
                      "trim(Valve_Type) as Valve_Type," \
                      "trim(Bearing_Vendor) as Bearing_Vendor," \
                      "trim(Fuel_Type) as Fuel_Type," \
                      "trim(Compression_ratio) as Compression_ratio," \
                      "trim(cam_arrangement) as cam_arrangement," \
                      "trim(Cylinder_arragement) as Cylinder_arragement," \
                      "trim(Turbocharger) as Turbocharger," \
                      "trim(Cylinder_deactivation) as Cylinder_deactivation," \
                      "trim(Direct_injection) as Direct_injection," \
                      "trim(displacement) as displacement," \
                      "trim(piston_type) as piston_type," \
                      "trim(Max_Torque) as Max_Torque," \
                      "trim(Peak_Power) as Peak_Power," \
                      "trim(Crankshaft_Design) as Crankshaft_Design," \
                      "trim(Liner_Design) as Liner_Design" \
                      ", case when TestA is null and TestA is null then 2 " \
                      " when TestA is not null and TestA is null then 3  " \
                      "when TestA is null and TestA is not null then 3  " \
                      "when TestA is not null and TestA is not null then 1 end as test_results" \
                      " from train1DF a LEFT JOIN trainAdditionalDF b ON (TestA = ID OR TestB = ID)"

newTrainDF = spark.sql(v_sql).persist().coalesce(1)

v_sql = " select DISTINCT trim(ID) as ID," \
        "case when trim(Number_of_Cylinders) is NULL or trim(Number_of_Cylinders) = '' or trim(Number_of_Cylinders) = ' ' or trim(Number_of_Cylinders) = 'NA' then " + (
            avgNoc) + " else trim(Number_of_Cylinders) end as Number_of_Cylinders," \
                      "trim(material_grade) as material_grade," \
                      "trim(Lubrication) as Lubrication," \
                      "trim(Valve_Type) as Valve_Type," \
                      "trim(Bearing_Vendor) as Bearing_Vendor," \
                      "trim(Fuel_Type) as Fuel_Type," \
                      "trim(Compression_ratio) as Compression_ratio," \
                      "trim(cam_arrangement) as cam_arrangement," \
                      "trim(Cylinder_arragement) as Cylinder_arragement," \
                      "trim(Turbocharger) as Turbocharger," \
                      "trim(Cylinder_deactivation) as Cylinder_deactivation," \
                      "trim(Direct_injection) as Direct_injection," \
                      "trim(displacement) as displacement," \
                      "trim(piston_type) as piston_type," \
                      "trim(Max_Torque) as Max_Torque," \
                      "trim(Peak_Power) as Peak_Power," \
                      "trim(Crankshaft_Design) as Crankshaft_Design," \
                      "trim(Liner_Design) as Liner_Design" \
                      ", case when TestA is null and TestA is null then 2 " \
                      " when TestA is not null and TestA is null then 3  " \
                      "when TestA is null and TestA is not null then 3  " \
                      "when TestA is not null and TestA is not null then 1 end as test_results" \
                      " from train1DF a LEFT JOIN trainAdditionalDF b ON (TestA = ID OR TestB = ID)"

newTestDF = spark.sql(v_sql).persist()

colNames = newTrainDF.columns
newTrainDF = f_fix_nulls_nas(colNames, newTrainDF)

colNames = newTestDF.columns
newTestDF = f_fix_nulls_nas(colNames, newTestDF)

testDF = newTestDF.withColumn("test_results", col('test_results').cast("Decimal")) \
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
            "Cylinder_deactivation",
            "Direct_injection",
            "displacement",
            "piston_type",
            "Max_Torque",
            "Peak_Power",
            "Crankshaft_Design",
            "Liner_Design",
            "test_results")

newTrainDF = spark.read.csv(
    dirVal + '\\' + "R_Train_Date_Imputed.csv",
    header=True, inferSchema=True) \
    .cache()

# print(  i for i in newTrainDF.stat.collect())

data_df = newTrainDF.withColumn("label", col('label').cast("Decimal")) \
    .withColumn("test_results", col('test_results').cast("Decimal")) \
    .withColumn("Number_of_Cylinders", col('Number_of_Cylinders').cast("Decimal")) \
    .select("label",
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
            "Cylinder_deactivation",
            "Direct_injection",
            "displacement",
            "piston_type",
            "Max_Torque",
            "Peak_Power",
            "Crankshaft_Design",
            "Liner_Design",
            "test_results")

data_df.coalesce(1).write.option("header", True).mode('overwrite').save("E:\\Insofe_BigData\\PHD\\PART2\\finalDataset.csv")

# 'Number_of_Cylinders',
numeric_cols = ['Number_of_Cylinders', 'test_results']
stringCols = ['material_grade', 'Lubrication', 'Valve_Type', 'Bearing_Vendor', 'Fuel_Type', 'Compression_ratio',
              'cam_arrangement', 'Cylinder_arragement', 'Turbocharger',
              'Cylinder_deactivation', 'Direct_injection',
              'displacement', 'piston_type', 'Max_Torque', 'Peak_Power', 'Crankshaft_Design',
              'Liner_Design']


def create_category_vars(dataset, field_name):
    # Create a new column with the suffix "Index" for each variable
    Index_col = field_name + "Index"
    # Create a new column with the suffix "Vec" for each variable
    Column_vec = field_name + "Vec"

    # For each variable return the index corresponding to the value in that variable
    # Define the StringIndex Object
    col_stringIndexer = StringIndexer(inputCol=field_name, outputCol=Index_col)
    # Find the no of indexes for that variable
    model = col_stringIndexer.fit(dataset)
    # Determine and return the index corresponding to the value in that variables into the columns column_name _ 'Index'
    idx_data = model.transform(dataset)

    # Using the Indexes returned from StringIndexer build and return the Vector of values for each variable
    encoder = OneHotEncoder(dropLast=True,
                            inputCol=Index_col,
                            outputCol=Column_vec)

    return encoder.transform(idx_data)


# COMMAND ----------

# Iterate over all the variables in the categorical variables list and run the function that is defined in the previous step
# on each of the variable - This function will return new columns appended to the existing dataframe.
for col in stringCols:
    data_df = create_category_vars(data_df, col)

# Verify the columns after OneHotEncoder
rows = data_df.count()
cols = len(data_df.columns)

cat_vecs = ["".join((cat, "Vec")) for cat in stringCols]
mod_features = numeric_cols + cat_vecs
assembler = VectorAssembler(inputCols=mod_features, outputCol="features")
data_df = assembler.transform(data_df)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scalerModel = scaler.fit(data_df)
data_df = scalerModel.transform(data_df)
print("count=>", data_df.count)
data_df[['features']].show(10)
data_df = data_df.withColumn("label", data_df['label'])
data_df.select("features", "label").show(5)

seed = 16578
(train_df, test_df) = data_df.randomSplit([0.8, 0.2], seed=seed)

evaluator = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName(
    "accuracy")

lr = LogisticRegression(maxIter=100, featuresCol="scaledFeatures", labelCol="label")

rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label", predictionCol="prediction",
                            probabilityCol="probability", seed=seed, subsamplingRate=0.0002)

pipeline = Pipeline(stages=[rf])
# .addGrid(rf.featureSubsetStrategy, ["auto", "log2"]) \
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [200, 100]) \
    .addGrid(rf.impurity, ["entropy"]) \
    .addGrid(rf.maxDepth, [30]) \
    .addGrid(rf.maxBins, [20, 10]) \
    .addGrid(rf.labelCol, ["label"]) \
    .addGrid(rf.featuresCol, ["features"]) \
    .build()

cvRF = call_crossValidator(train_df, test_df, paramGrid, pipeline, evaluator)

# print(cvRF.featureImportances)

pipeline = Pipeline(stages=[lr])
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.threshold, [0.6, 0.4]) \
    .build()

cvLR = call_crossValidator(train_df, test_df, paramGrid, pipeline, evaluator)

gbt = GBTClassifier(featuresCol="features", labelCol="label", predictionCol="prediction",
                    seed=seed, subsamplingRate=0.02)

pipeline = Pipeline(stages=[gbt])
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [30]) \
    .addGrid(gbt.labelCol, ["label"]) \
    .addGrid(gbt.featuresCol, ["features"]) \
    .build()

cvGBT = call_crossValidator(train_df, test_df, paramGrid, pipeline, evaluator)
# Actual Test
print("Actual Test")
fnl_numeric_cols = ['Number_of_Cylinders', 'test_results']
fnl_stringCols = ['ID', 'material_grade', 'Lubrication', 'Valve_Type', 'Bearing_Vendor', 'Fuel_Type',
                  'Compression_ratio',
                  'cam_arrangement', 'Cylinder_arragement', 'Turbocharger',
                  'Cylinder_deactivation', 'Direct_injection',
                   'displacement', 'piston_type', 'Max_Torque', 'Peak_Power', 'Crankshaft_Design',
                  'Liner_Design']

for col in fnl_stringCols:
    testDF = create_category_vars(testDF, col)

cat_vecs = ["".join((cat, "Vec")) for cat in fnl_stringCols]
mod_features = fnl_numeric_cols + cat_vecs
assembler = VectorAssembler(inputCols=mod_features, outputCol="features")
testDF = assembler.transform(testDF)

actual_test_predictions_rf = cvRF.transform(testDF)
testResults_save = actual_test_predictions_rf.select("ID", "prediction", "probability").coalesce(1)


# testResults_save.toDF().coalesce(1).write.option("header", True).save("E:\\Insofe_BigData\\PHD\\PART2\\testResults_save2.csv")

def run_nlp(train_df, test_df):
    # len(features)
    # layers = [21, 80, 3]
    layers = [train_df.schema["features"].metadata["ml_attr"]["num_attrs"], 160, 150, 50, 10, 2]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(featuresCol='features', labelCol='label', predictionCol='prediction',
                                             maxIter=200, layers=layers, stepSize=0.0003, blockSize=30, tol=0.00001,
                                             seed=seed)
    # train the model
    model = trainer.fit(train_df)
    # compute accuracy on the test set
    predictTest = model.transform(test_df)
    predictionAndLabels = predictTest.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy of MLP = " + str(evaluator.evaluate(predictionAndLabels)))
    return model


mlp_model = run_nlp(train_df, test_df)

exit(0)
