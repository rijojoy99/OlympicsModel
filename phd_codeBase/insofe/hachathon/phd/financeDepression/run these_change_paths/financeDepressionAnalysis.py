from __future__ import print_function
from io import open
import sys
import pyspark.sql.functions as fn
# from pyspark.sql.functions import avg, col, when
from pyspark.sql import Window
from pyspark.sql import column
from pyspark.sql import SparkSession
import numpy as NP

spark = SparkSession.builder.appName("FinDep").getOrCreate()

dirVal = "/user/1533B33/B33PHD" + sys.argv[1]
#dirVal = "E:\\Insofe_BigData\\PHD"
filename =  sys.argv[2]
# "Batch33_phdData.csv"
#


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

prctDist.show()

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


newFinDepDF.show(20)

# meanAge = newFinDepDF.select(newFinDepDF.columns[])

newFinDepDF.select(newFinDepDF.columns[9]).fillna(0)

newFinDepDF.select(newFinDepDF.columns[0:5]).fillna('mean')

newFinDepDF.show(10)

# modeCols = NP.Seq("Utilization","Age","Debt_Ratio","Monthly_Income")
# mean_dict = { col: 'mean' for col in meanCols }
#
# col_avgs = newFinDepDF.agg( mean_dict ).collect()[0].asDict()
# col_avgs = { k[4:-1]: v for k,v in col_avgs.iteritems() }
# newFinDepDF.fillna( col_avgs ).show()

#


# print(initFinDep.schema)
# # Find the total number of medals won by each country in hockey.
# tot_cntry = spark.createDataFrame(initFinDep.repartition(initFinDep['country']).filter(" game_category = 'Hockey' ") \
#     .rdd.map(lambda x: ((x[2]), (x[6] + x[7] + x[8]))).reduceByKey(
#     lambda x, y: x + y)).toDF('country', 'num_of_medals')
# # Find the number of medals that India won in each Olympics
# olympics_IndiaMedals = spark.createDataFrame(
#     initFinDep.repartition(initFinDep['country']).filter(" country = 'India' ") \
#         .rdd.map(lambda x: ((x[2]), (x[6] + x[7] + x[8]))).reduceByKey(lambda x, y: x + y)).toDF('country',
#                                                                                                  'num_of_medals_india')
# # Find the total number of medals won by each country.
# tot_num_medals_cntry = spark.createDataFrame(initFinDep.repartition(initFinDep['country']) \
#     .rdd.map(lambda x: ((x[2]), (x[6] + x[7] + x[8]))).reduceByKey(
#     lambda x, y: x + y)).toDF('country', 'num_of_medals_cntry')
# # Find the total number of medals won by each country and in each Olympics.
# tot_num_medals_cntry_olympics = spark.createDataFrame(
#     initFinDep.repartition(initFinDep['country'], initFinDep['olympic_year']) \
#         .rdd.map(lambda x: ((x[2], x[3]), (x[6] + x[7] + x[8]))).reduceByKey(lambda x, y: x + y).map(
#         lambda x: (x[0][0], x[0][1], x[1]))) \
#     .withColumnRenamed("_1", "country").withColumnRenamed("_2", "olympic_year").withColumnRenamed("_3",
#                                                                                                   "num_of_medals_cntry")
#
# print(tot_num_medals_cntry_olympics.collect())
# tot_num_medals_cntry_olympics.foreach(print)
#
# # Find all the athletes from India in each Olympics.
# olympics_IndiaMedals = spark.createDataFrame(
#     initFinDep.repartition(initFinDep['country']).filter(" country = 'India' ") \
#         .rdd.map(lambda x: (x[0],))).distinct().toDF('atheletes')
# #
# #
# # tot_cntry.foreach(print)
# # olympics_IndiaMedals.foreach(print)
# # tot_num_medals_cntry.foreach(print)
# # tot_num_medals_cntry_olympics.foreach(print)
# # olympics_IndiaMedals.foreach(print)
# #
#
# tot_num_medals_cntry.coalesce(1).write.format('com.databricks.spark.csv').option("header", True).mode(
#     saveMode="Overwrite").csv(dirVal + "/" + "tot_num_medals_cntry.csv")
# tot_cntry.coalesce(1).write.format('com.databricks.spark.csv').option("header", True).mode(saveMode="Overwrite").csv(
#     dirVal + "/" + "tot_cntry.csv")
# tot_num_medals_cntry_olympics.coalesce(1).write.format('com.databricks.spark.csv').option("header", True).mode(
#     saveMode="Overwrite").csv(dirVal + "/" + "tot_num_medals_cntry_olympics.csv")
# olympics_IndiaMedals.coalesce(1).write.format('com.databricks.spark.csv').option("header", True).mode(
#     saveMode="Overwrite").csv(dirVal + "/" + "olympics_IndiaMedals.csv")
#
exit(0)
