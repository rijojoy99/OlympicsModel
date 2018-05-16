from __future__ import print_function
from io import open
import sys
from pyspark.sql.functions import StringType
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Olympics").getOrCreate()

dirVal = "/user/1533B33/" + sys.argv[1]
filename = sys.argv[2]

initOlympics = spark.read.csv(
    dirVal + '/' + filename,
    header=False, inferSchema=True) \
    .toDF("athele_name", "age", "country", "olympic_year", "date_of_game", "game_category", "gold", "silver", "bronze",
          "total_medals").cache()

print(initOlympics.schema)
# Find the total number of medals won by each country in hockey.
tot_cntry = spark.createDataFrame(initOlympics.repartition(initOlympics['country']).filter(" game_category = 'Hockey' ") \
    .rdd.map(lambda x: ((x[2]), (x[6] + x[7] + x[8]))).reduceByKey(
    lambda x, y: x + y)).toDF('country', 'num_of_medals')
# Find the number of medals that India won in each Olympics
olympics_IndiaMedals = spark.createDataFrame(
    initOlympics.repartition(initOlympics['country']).filter(" country = 'India' ") \
        .rdd.map(lambda x: ((x[2]), (x[6] + x[7] + x[8]))).reduceByKey(lambda x, y: x + y)).toDF('country',
                                                                                                 'num_of_medals_india')
# Find the total number of medals won by each country.
tot_num_medals_cntry = spark.createDataFrame(initOlympics.repartition(initOlympics['country']) \
    .rdd.map(lambda x: ((x[2]), (x[6] + x[7] + x[8]))).reduceByKey(
    lambda x, y: x + y)).toDF('country', 'num_of_medals_cntry')
# Find the total number of medals won by each country and in each Olympics.
tot_num_medals_cntry_olympics = spark.createDataFrame(
    initOlympics.repartition(initOlympics['country'], initOlympics['olympic_year']) \
        .rdd.map(lambda x: ((x[2], x[3]), (x[6] + x[7] + x[8]))).reduceByKey(lambda x, y: x + y).map(
        lambda x: (x[0][0], x[0][1], x[1]))) \
    .withColumnRenamed("_1", "country").withColumnRenamed("_2", "olympic_year").withColumnRenamed("_3",
                                                                                                  "num_of_medals_cntry")

print(tot_num_medals_cntry_olympics.collect())
tot_num_medals_cntry_olympics.foreach(print)

# Find all the athletes from India in each Olympics.
olympics_IndiaMedals = spark.createDataFrame(
    initOlympics.repartition(initOlympics['country']).filter(" country = 'India' ") \
        .rdd.map(lambda x: (x[0], ))).distinct().toDF('atheletes')
#
#
# tot_cntry.foreach(print)
# olympics_IndiaMedals.foreach(print)
# tot_num_medals_cntry.foreach(print)
# tot_num_medals_cntry_olympics.foreach(print)
# olympics_IndiaMedals.foreach(print)
#

tot_num_medals_cntry.coalesce(1).write.format('com.databricks.spark.csv').option("header",True).mode(saveMode="Overwrite").csv(dirVal + "/" + "tot_num_medals_cntry.csv")
tot_cntry.coalesce(1).write.format('com.databricks.spark.csv').option("header",True).mode(saveMode="Overwrite").csv(dirVal + "/" + "tot_cntry.csv")
tot_num_medals_cntry_olympics.coalesce(1).write.format('com.databricks.spark.csv').option("header",True).mode(saveMode="Overwrite").csv(dirVal + "/" + "tot_num_medals_cntry_olympics.csv")
olympics_IndiaMedals.coalesce(1).write.format('com.databricks.spark.csv').option("header",True).mode(saveMode="Overwrite").csv(dirVal + "/" + "olympics_IndiaMedals.csv")

exit(0)
