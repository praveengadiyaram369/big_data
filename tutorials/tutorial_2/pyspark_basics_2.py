from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, col
from pyspark.sql.functions import mean, min, max, stddev_pop, stddev_samp, covar_pop, corr
from pyspark.sql.functions import sum

# _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
sc = SparkContext("local", "Hello World")
sc.setLogLevel('ERROR')

# _start spark sessin from context
spark = SparkSession(sc)

# _read csv file
flightdata = spark.read.option('inferSchema', 'true').option(
    'header', 'true').csv('2015-summary.csv')
flightdata.show(5)
flightdata.printSchema()

# _add new column using withColumn, we are just printing the updated dataframe by show, but it should be taken into new variable as new dataframe.
flightdata.withColumn("newCol", col("count")+10).show(4)

# _using select, we can also mention column names explicitly in place of *
flightdata_mod = flightdata.select("*", (col("count")+20).alias("newCol2"))
flightdata_mod.show(5)

# _basic statistical functions
flightdata.select(mean("count")).show()
flightdata.select(min("count")).show()
flightdata.select(max("count")).show()
flightdata.select(stddev_pop("count")).show()
flightdata.select(stddev_samp("count")).show()
flightdata.select()

# _group by and aggregations
flightdata.groupBy("DEST_COUNTRY_NAME").agg(sum('count')).show(5)
dest_count_data = flightdata.groupBy(
    "DEST_COUNTRY_NAME").agg({'count': 'sum'})

# _write the data to csv after coalesce
dest_count_data_merged = dest_count_data.coalesce(1)
dest_count_data_merged.write.format('csv').option(
    'header', 'true').save('dest_country')
