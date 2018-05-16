# Necessary Imports
from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

# Define Schema
sch    = StructType([
         StructField("store_id", IntegerType(), True),
         StructField("store_type", StringType(), True),
         StructField("assortment_level", StringType(), True),
         StructField("date", DateType(), True),
         StructField("competitor_distance", DoubleType(), True),
         StructField("competitor_months", DoubleType(), True),
         StructField("day_of_week", IntegerType(), True),
         StructField("month", IntegerType(), True),
         StructField("year", IntegerType(), True),
         StructField("sale_amount", DoubleType(), True),
         StructField("customer_count", IntegerType(), True),
         StructField("open", IntegerType(), True), 
         StructField("promo", IntegerType(), True),
         StructField("state_holiday", StringType(), True),
         StructField("school_holiday", StringType(), True),
         StructField("next_day_state_holiday", StringType(), True),
         StructField("next_day_school_holiday", IntegerType(), True),
         StructField("avg_sales_last_2days", DoubleType(), True),
         StructField("avg_sales_last_3days", DoubleType(), True),
         StructField("avg_sales_last_4days", DoubleType(), True),
         StructField("avg_sales_last_5days", DoubleType(), True),
         StructField("avg_sales_by_month", DoubleType(), True)])

# COMMAND ----------

# Construct Dataframe
sales_DF = sqlContext.read.format("com.databricks.spark.csv").options(
                             delimiter = ',', 
                             inferschema = True,
                             header = False).load("<path>",schema = sch)

# COMMAND ----------

# Filter records where the sale_amount > 0
data_df = sales_DF.where( sales_DF.sale_amount > 0 )

# COMMAND ----------

# Fill Null Values with Zeroes
data_df=data_df.fillna(0)

# COMMAND ----------

# Drop/Delete Null values if any
data_df = data_df.na.drop( how = 'any' )

# COMMAND ----------

# Define the list of the categorical variables - To dummy
categorical_list1 = ['day_of_week',
                     'promo',
                     'state_holiday',
                     'school_holiday',
                     'month',
                     'year',
                     'store_type',
                     'assortment_level']

# COMMAND ----------

# Define the list of the continuous variables 
continuous_list1 = ['competitor_distance', 'competitor_months']

# COMMAND ----------

# Necessary imports for dummy
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, PolynomialExpansion, VectorIndexer

# COMMAND ----------

# Verify the dataset before the dummy - Verify the columns
data_df.show(10)

# COMMAND ----------

# Verify the columns before OneHotEncoder
rows = data_df.count()
cols = len(data_df.columns)           
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
  model = col_stringIndexer.fit( data_df )
  # Determine and return the index corresponding to the value in that variables into the columns column_name _ 'Index'
  idx_data = model.transform( data_df )

  # Using the Indexes returned from StringIndexer build and return the Vector of values for each variable
  encoder = OneHotEncoder( dropLast=True,
                                 inputCol=Index_col,
                                 outputCol= Column_vec )

  return encoder.transform( idx_data )

# COMMAND ----------

# Iterate over all the variables in the categorical variables list and run the function that is defined in the previous step
# on each of the variable - This function will return new columns appended to the existing dataframe.
for col in categorical_list1:
  data_df = create_category_vars( data_df, col )

# Verify the columns after OneHotEncoder
rows = data_df.count()
cols = len(data_df.columns)           
print('After OneHotEncoding, Total No of Rows and Columns are {} rows and {} columns'.format(rows, cols))

# COMMAND ----------

# Make the dataframe available in-memory
data_df.cache()

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
data_df = assembler.transform(data_df)

# COMMAND ----------

# Verify the columns after VectorAssember
rows = data_df.count()
cols = len(data_df.columns)           
print('After VectorAssembler, Total No of Rows and Columns are {} rows and {} columns'.format(rows, cols))

# COMMAND ----------

# Verify the newly created columns 'features' in the dataframe
# This returns sparse vector
data_df[['features']].show(10)

# COMMAND ----------

# Rename the target variable as label
data_df = data_df.withColumn( "label", data_df.sale_amount )

# COMMAND ----------

# Verify the Independent (features) and Dependent (Label) fields
data_df.select( "features", "label" ).show( 5 )

# COMMAND ----------

# Split the data as train and test sets
seed = 16578
train_df, test_df = data_df.randomSplit( [0.7, 0.3], seed = seed )

# COMMAND ----------

# Build a linear regression model
from pyspark.ml.regression import LinearRegression
linreg = LinearRegression(maxIter=500, regParam=0.0)
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

# COMMAND ----------

#### Add More features to improve the model

# Filter records where the sale_amount > 0
new_data_df = sales_DF.where( sales_DF.sale_amount > 0 )

# Fill Null Values with Zeroes
new_data_df =new_data_df.fillna(0)

# Drop/Delete Null values if any
new_data_df = new_data_df.na.drop( how = 'any' )

# COMMAND ----------

# Define the list of the categorical variables - To dummy
categorical_list2 = ['day_of_week',
                     'promo',
                     'state_holiday',
                     'school_holiday',
                     'month',
                     'year',
                     'store_type',
                     'assortment_level',
                     'next_day_state_holiday',
                     'next_day_school_holiday']

# COMMAND ----------

# Define the list of the continuous variables
continuous_list2 = ['competitor_distance', 'competitor_months','customer_count','avg_sales_last_2days', 'avg_sales_last_3days','avg_sales_last_4days','avg_sales_last_5days','avg_sales_by_month']

# COMMAND ----------

# Verify the dataset before the dummy - Verify the columns
new_data_df.show(10)

# COMMAND ----------

# Verify the columns before OneHotEncoder
rows = new_data_df.count()
cols = len(new_data_df.columns)           
print('Before OneHotEncoding, Total No of Rows and Columns are {} rows and {} columns'.format(rows, cols))

# COMMAND ----------

# Define the function for OneHotEncoding
# OneHot Encoding is two step process
# 1. Find the indexes for the categories using StringIndexer
# 2. Using the above index do onehot encoding
def create_category_vars( dataset, field_name ):
  # Create a new column with the suffix "Index" for each variable
  Index_col = field_name + "_Index"
  # Create a new column with the suffix "Vec" for each variable
  Column_vec = field_name + "_Vec"

  # For each variable return the index corresponding to the value in that variable
  # Define the StringIndex Object
  col_stringIndexer = StringIndexer(inputCol=field_name, outputCol=Index_col )
  # Find the no of indexes for that variable
  model = col_stringIndexer.fit( new_data_df )
  # Determine and return the index corresponding to the value in that variables into the columns column_name _ 'Index'
  idx_data = model.transform( new_data_df )

  # Using the Indexes returned from StringIndexer build and return the Vector of values for each variable
  encoder = OneHotEncoder( dropLast=True,
                                 inputCol=Index_col,
                                 outputCol= Column_vec )

  return encoder.transform( idx_data )

# COMMAND ----------

# Iterate over all the variables in the categorical variables list and run the function that is defined in the previous step
# on each of the variable - This function will return new columns appended to the existing dataframe.
for col in categorical_list2:
  new_data_df = create_category_vars( new_data_df, col )

# COMMAND ----------

# Verify the columns after OneHotEncoder
rows = new_data_df.count()
cols = len(new_data_df.columns)           
print('After OneHotEncoding, Total No of Rows and Columns are {} rows and {} columns'.format(rows, cols))

# COMMAND ----------

# Make the dataframe available in-memory
new_data_df.cache()

# COMMAND ----------

# A list that contains all the vector variable names that are created during onehot encoder
cat_vecs = [ "".join( (cat, "_Vec") ) for cat in categorical_list2 ]
# Create a list with the final set of variables for the model building
# Both Continuous variables and dummy variables (vectors created in the onehot encoding)
mod_features = continuous_list2 + cat_vecs
mod_features

# COMMAND ----------

# Merge values from all the features/variables/attributes into a single vector
# Using VectorAssembler
# Here, 
# input is set of numeric and vectors(dummys)
# Output is a new column (features in this case) 
assembler = VectorAssembler( inputCols = mod_features, outputCol = "features")
new_data_df = assembler.transform(new_data_df)

# COMMAND ----------

# Rename the target variable as label
new_data_df = new_data_df.withColumn( "label", new_data_df.sale_amount )
# Verify the Independent (features) and Dependent (Label) fields
new_data_df.select( "features", "label" ).show( 5 )

# COMMAND ----------

# Split the data as train and test sets
seed = 100
train_df, test_df = new_data_df.randomSplit( [0.7, 0.3], seed = seed )

# COMMAND ----------

# Build a linear regression model
lr = LinearRegression(maxIter=500, regParam=0.0)
lr_new = linreg.fit( train_df )

# COMMAND ----------

print('Intercept is :', lr_new.intercept)
print('Coefficients are: ', lr_new.coefficients)

# COMMAND ----------

# Predictions on Test Data
pred = lr_new.transform( test_df )
# Select Features Vector, Actual Value and Predicted value
pred.select( 'features', 'label', 'prediction' ).show( 5 )

# COMMAND ----------

# Find the error metric - RMSE
from pyspark.ml.evaluation import RegressionEvaluator
rmse = RegressionEvaluator(labelCol="label",
                            predictionCol="prediction",
                            metricName="rmse" )

lm_rmse = rmse.evaluate( pred)
lm_rmse

# COMMAND ----------

# Find R2 value
rsquared = RegressionEvaluator(labelCol="label",
                            predictionCol="prediction",
                            metricName="r2" )
r2 = rsquared.evaluate( pred )
r2
