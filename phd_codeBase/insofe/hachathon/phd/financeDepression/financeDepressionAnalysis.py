from __future__ import print_function
from io import open
import sys
import pyspark.sql.functions as fn
from pyspark.sql.functions import  col, when
from pyspark.sql import Window
from pyspark.sql import column
from pyspark.sql import SparkSession
import numpy as NP

spark = SparkSession.builder.appName("FinDep").getOrCreate()

# dirVal = "/user/1533B33/B33PHD" + sys.argv[1]
dirVal = "E:\\Insofe_BigData\\PHD"
filename = "Batch33_phdData.csv"
# sys.argv[2]


# Read this data and create a data frame and verify the dataframe.
initFinDep = spark.read.csv(
    dirVal + '\\' + filename,
    header=False, inferSchema=True) \
    .toDF("Target",
          "Utilization",
          "age",
          "FD_ind1",
          "Debt_Ratio",
          "Monthly_Income",
          "FD_ind2",
          "FD_ind3",
          "FD_ind4",
          "FD_ind5",
          "NumberOfDependents").cache()

dfSchema = initFinDep.printSchema
initFinDep.show(10)
total = int(initFinDep.count())
# Display the count of rows and columns.
print("Total rows=>", total, "\n", "Total Columns =>", initFinDep.columns.__len__())

# Give the percentage distribution of Target attribute and verify if it is a class imbalance problem or not.
prctDist = initFinDep.groupBy('Target').count() \
    .withColumn('total', fn.lit(total)) \
    .withColumn('Percent', fn.expr('count/total'))

# class Imbalance more of zeroes
# +------+------+------+--------+
# |Target| count| total|fraction|
# +------+------+------+--------+
# |     1| 10026|150000| 0.06684|
# |     0|139974|150000| 0.93316|
# +------+------+------+--------+

count_of_na = initFinDep.na.__sizeof__()

print("Count of NAs =>", count_of_na)

#
# After you create the dataframe in the first step, Target attribute will be in the first
# column of the dataframe, make it as the last column of the dataframe.

newFinDepDF = initFinDep.selectExpr(
    "Utilization",
    "age",
    "FD_ind1",
    "Debt_Ratio",
    "Monthly_Income",
    "FD_ind2",
    "FD_ind3",
    "FD_ind4",
    "FD_ind5",
    "NumberOfDependents", "Target")

newFinDepDF.show(10)

print(newFinDepDF.describe)

# newFinDepDF.columns agg(avg(col("age")))
# newFinDepDF.withColumn("avgUtil", (avg(col("Utilization")))) \
# .withColumn("avgAge", fn.avg(['Age']))\
#     .withColumn("FD_ind1Mode",fn.mode(['FD_ind1']))\
#     .withColumn("DebtRatioAvg", fn.avg(['Debt_Ratio']))\
# .withColumn("Monthly_IncomeAvg", fn.avg(['Monthly_Income']))
#
newFinDepDF.select(newFinDepDF.columns[9]).fillna(0 )

# newFinDepDF.select(newFinDepDF.columns[0:4]).fillna(newFinDepDF.agg(column[0:4]))
# AgeMean = select(agg(avg(newFinDepDF.columns[0])) )

AvgCols = ['age',
           'Utilization',
           'FD_ind1',
           'Debt_Ratio',
           'Monthly_Income']

for i in AvgCols:
    newFinDepDF.rdd.map(lambda x: x.fillna(x.mean()))

newFinDepDF.show(20)

ModeCols = newFinDepDF.columns

for i in ModeCols:
    newFinDepDF.rdd.map(lambda x: x.fillna(x.mode()))

newFinDepDF.fillna(0)\
    .show(10)

newFinDepDF.createOrReplaceTempView("tmp_initFinDep")

newDF = spark.sql(" select avg(k.Utilization) over (order by 1) as avg_util,"
                  " avg(k.Utilization) over (order by 1) as avg_util,"
                  "avg(k.Utilization) over (order by 1) as avg_util,"
                  "avg(k.Utilization) over (order by 1) as avg_util,"
                  "avg(k.Utilization) over (order by 1) as avg_util,"
                  "avg(k.Utilization) over (order by 1) as avg_util,"
                  "avg(k.Utilization) over (order by 1) as avg_util,"
                  "k.*"
                  " from tmp_initFinDep k "
                  ""
                  "limit 100")
newDF.show()

exit(1)


# Define the list of the categorical variables - To dummy
categorical_list1 = ['age',
           'Utilization',
           'FD_ind1',
           'Debt_Ratio',
           'Monthly_Income']

# COMMAND ----------

# Define the list of the continuous variables
continuous_list1 = ['competitor_distance', 'competitor_months']

# COMMAND ----------

# Necessary imports for dummy
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, PolynomialExpansion, VectorIndexer

# COMMAND ----------

# Verify the dataset before the dummy - Verify the columns
newFinDepDF.show(10)

# COMMAND ----------

# Verify the columns before OneHotEncoder
rows = newFinDepDF.count()
cols = len(newFinDepDF.columns)
print('Before OneHotEncoding, Total No of Rows and Columns are {} rows and {} columns'.format(rows, cols))

# COMMAND ----------

# Define the function for OneHotEncoding
# OneHot Encoding is two step process
# 1. Find the indexes for the categories using StringIndexer
# 2. Using the above index do onehot encoding
def create_category_vars( dataset, field_name ):
  # Create a new column with the suffix "Index" for each variable
  Index_col = field_name + "Index"
  # Create a new column with the suffix "Vec" for each variable
  Column_vec = field_name + "Vec"

  # For each variable return the index corresponding to the value in that variable
  # Define the StringIndex Object
  col_stringIndexer = StringIndexer(inputCol=field_name, outputCol=Index_col )
  # Find the no of indexes for that variable
  model = col_stringIndexer.fit( newFinDepDF )
  # Determine and return the index corresponding to the value in that variables into the columns column_name _ 'Index'
  idx_data = model.transform( newFinDepDF )

  # Using the Indexes returned from StringIndexer build and return the Vector of values for each variable
  encoder = OneHotEncoder( dropLast=True,
                                 inputCol=Index_col,
                                 outputCol= Column_vec )

  return encoder.transform( idx_data )

# COMMAND ----------

# Iterate over all the variables in the categorical variables list and run the function that is defined in the previous step
# on each of the variable - This function will return new columns appended to the existing dataframe.
for col in categorical_list1:
  newFinDepDF = create_category_vars( newFinDepDF, col )

# Verify the columns after OneHotEncoder
rows = newFinDepDF.count()
cols = len(newFinDepDF.columns)
print('After OneHotEncoding, Total No of Rows and Columns are {} rows and {} columns'.format(rows, cols))

# COMMAND ----------

# Make the dataframe available in-memory
newFinDepDF.cache()

# COMMAND ----------

# A list that contains all the vector variable names that are created during onehot encoder
cat_vecs = [ "".join( (cat, "Vec") ) for cat in categorical_list1]

# COMMAND ----------

# Create a list with the final set of variables for the model building
# Both Continuous variables and dummy variables (vectors created in the onehot encoding)
mod_features = continuous_list1 + cat_vecs

# COMMAND ----------

# Merge values from all the features/variables/attributes into a single vector
# Using VectorAssembler
# Here,
# input is set of numeric and vectors(dummys)
# Output is a new column (features in this case)
assembler = VectorAssembler( inputCols = mod_features, outputCol = "features")
newFinDepDF = assembler.transform(newFinDepDF)

# COMMAND ----------

# Verify the columns after VectorAssember
rows = newFinDepDF.count()
cols = len(newFinDepDF.columns)
print('After VectorAssembler, Total No of Rows and Columns are {} rows and {} columns'.format(rows, cols))

# COMMAND ----------

# Verify the newly created columns 'features' in the dataframe
# This returns sparse vector
newFinDepDF[['features']].show(10)

# COMMAND ----------

# Rename the target variable as label
newFinDepDF = newFinDepDF.withColumn( "label", newFinDepDF.sale_amount )

# COMMAND ----------

# Verify the Independent (features) and Dependent (Label) fields
newFinDepDF.select( "features", "label" ).show( 5 )

# COMMAND ----------

# Split the data as train and test sets
seed = 16578
train_df, test_df = newFinDepDF.randomSplit( [0.7, 0.3], seed = seed )

# COMMAND ----------

# Build a linear regression model
from pyspark.ml.classification import LogisticRegression
linreg = LogisticRegression(maxIter=500)
lm = linreg.fit( train_df )

# COMMAND ----------

print('Intercept is :', lm.intercept)
print('Coefficients are: ', lm.coefficients)

# COMMAND ----------

# Predictions on Test Data
pred = lm.transform( test_df )
# Select Features Vector, Actual Value and Predicted value
pred.select( 'features', 'label', 'prediction' ).show( 5 )

# COMMAND ----------

# Find the error metric - RMSE
from pyspark.ml.evaluation import RegressionEvaluator
rmse = RegressionEvaluator(labelCol="label",
                            predictionCol="prediction",
                            metricName="rmse" )
lm_rmse = rmse.evaluate( pred)
print('RMSE value on Test data is', lm_rmse)

# COMMAND ----------

# Find R2 value
rsquared = RegressionEvaluator(labelCol="label",
                            predictionCol="prediction",
                            metricName="r2" )
r2 = rsquared.evaluate( pred )
r2


exit(0)
