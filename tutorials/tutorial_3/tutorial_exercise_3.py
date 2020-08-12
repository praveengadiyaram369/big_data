from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *

if __name__ == "__main__":

    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _start spark sessin from context
    spark = SparkSession(sc)

    # _read csv file
    retail_data = spark.read.option('inferSchema', 'true').option(
        'header', 'true').option('timestampFormat', 'dd/MM/yyyy H:mm').csv('online-retail-dataset.csv')

    # _1.spark sql query
    retail_data.select(hour(col("InvoiceDate")).alias("InvoiceHour"), "InvoiceNo").groupBy(col("InvoiceHour")).agg(
        countDistinct(col("InvoiceNo")).alias("InvoiceOrderCnt")).orderBy(col("InvoiceHour")).show()

    # _1.sql query
    retail_data.createOrReplaceTempView('retailTable')
    spark.sql('''select hour(InvoiceDate) as InvoiceHour,count(distinct InvoiceNo) as InvoiceOrderCnt  from retailTable group by InvoiceHour  order by InvoiceHour ''').show(5)

    # _2.spark sql query
    retail_data_selection = retail_data.select(
        "Country", "StockCode", "Quantity")
    retail_data_selection = retail_data_selection.na.replace(
        [""], ["UNKNOWN"], "StockCode").na.replace([""], ["UNKNOWN"], "Country").na.drop("any")
    retail_data_selection.createOrReplaceTempView('retailSelectionTable')

    result_data = retail_data_selection.cube("Country", "StockCode").agg(
        sum("Quantity").alias("productFrequency")).orderBy("Country", "StockCode")

    result_data.coalesce(1).write.format('csv').option(
        'header', 'true').save('retail_data_cube_result')

    # _2.sql query
    retail_data_selection.createOrReplaceTempView('retailSelectionTable')
    spark.sql(''' select Country, StockCode, sum(Quantity) as productFrequency from retailSelectionTable group by Country, StockCode grouping sets ((Country, StockCode), (Country), (StockCode), ()) order by Country, StockCode''').show(5)

    sc.stop()
