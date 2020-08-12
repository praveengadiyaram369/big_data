from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum

if __name__ == "__main__":
    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _start spark sessin from context
    spark = SparkSession(sc)

    # _read csv file
    flight_data = spark.read.option('inferSchema', 'true').option(
        'header', 'true').csv('2015-summary.csv')
    flight_data.show(5)
    flight_data.printSchema()
    flight_data.filter(col("DEST_COUNTRY_NAME") == "United States").groupBy("ORIGIN_COUNTRY_NAME").agg(
        sum("count").alias("origin_cnt")).orderBy("ORIGIN_COUNTRY_NAME").show(5)

    flight_data.createOrReplaceTempView('flightTable')
    spark.sql('''select ORIGIN_COUNTRY_NAME,sum(count) as origin_cnt  from flightTable where DEST_COUNTRY_NAME="United States" group by ORIGIN_COUNTRY_NAME  order by ORIGIN_COUNTRY_NAME ''').show(5)

    spark.stop()
