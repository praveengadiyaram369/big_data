from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import desc, col
from pyspark.sql.functions import mean, min, max, stddev_pop, stddev_samp, covar_pop, corr
from pyspark.sql.functions import sum, size, count

# _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
sc = SparkContext("local", "Hello World")
sc.setLogLevel('ERROR')

# _start spark sessin from context
spark = SparkSession(sc)

# _read csv file
retail_data = spark.read.option('inferSchema', 'true').option(
    'header', 'true').csv('online-retail-dataset.csv')
retail_data.show(5)
retail_data.printSchema()

# _most purchased item;; InvoiceNo -> StockCode -> Description -> Quantity
retail_data_most_purchased = retail_data.groupBy(
    "StockCode").agg(sum(col('Quantity')).alias('quantity_sum'))
retail_data_most_purchased.select(max("quantity_sum")).show()  # _56450
retail_data_most_purchased.select("StockCode").where(
    col("quantity_sum") == 56450).show()  # _22197
retail_data.select("StockCode", "Description").where(
    col("StockCode") == 22197).dropDuplicates().show()

# _most purchased item;; usa
retail_data_most_purchased_usa = retail_data.filter(col("Country") == "USA").groupBy(
    "StockCode").agg(sum(col('Quantity')).alias('quantity_sum'))
retail_data_most_purchased_usa.select(max("quantity_sum")).show()  # _88
retail_data_most_purchased_usa.select("StockCode").where(
    col("quantity_sum") == 88).show()  # _23366
retail_data.select("StockCode", "Description").where(
    col("StockCode") == 23366).dropDuplicates().show()

# _highest invoice
retail_data_invoice = retail_data.groupBy("InvoiceNo").agg(
    sum(col('Quantity') * col('UnitPrice')).alias('invoice_sum'))
retail_data_invoice.orderBy(desc("invoice_sum")).show(1)  # _581483

# _lowest invoice
retail_data_invoice.filter(col('invoice_sum') > 0).orderBy(
    "invoice_sum").show(1)  # _570554

# _adding new columns
retail_data = retail_data.withColumn("is_germany", col(
    "Country") == "Germany")
retail_data = retail_data.join(retail_data_invoice, ["InvoiceNo"])
retail_data.show(5)

# _count of german customers who purchased more than 10 dollars
retail_data.filter(col('is_germany') == 'true').groupBy(
    "InvoiceNo").agg(sum(col('Quantity') * col('UnitPrice')).alias('invoice_total_sum')).where(col("invoice_total_sum") > 10).select(count(col("InvoiceNo"))).show(1)  # _455

# _german customers with total invoice sum in descending order
retail_data.filter(col('is_germany') == 'true').groupBy(col("InvoiceNo"), col("CustomerID")
                                                        ).agg(sum(col('Quantity') * col('UnitPrice')).alias('invoice_total_sum')).orderBy(desc("invoice_total_sum")).show(5)

sc.stop()
