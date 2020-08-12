from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, col
from pyspark.sql.functions import sum

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

retail_selection = retail_data.select("Country", "StockCode", "Quantity")
retail_selection = retail_selection.groupBy(
    "StockCode", "Country").agg(sum("Quantity").alias("quantity_sum"))
retail_selection.show(5)

# _slice data - fix one dimension of the data - use either use where or filter on one column with a fixed value
retail_slice = retail_selection.where(col("Country") == "United Kingdom")
retail_slice.show(5)

# _dice data - remove some part of the data, but don't reduce the dimension by filtering one column with a fixed value
retail_dice = retail_selection.where((col("Country") == "United Kingdom") | (col(
    "Country") == "USA")).where((col("quantity_sum") == 0) | (col("quantity_sum") == 1))
retail_dice.show(5)

# _pivot the data
retail_pivot = retail_data.groupBy(
    "StockCode").pivot("Country").agg(sum("Quantity"))
retail_pivot.show(5)

# _roll-up data - follows a path in the lattice by always leaving out some columns - hierarchically from left to right where left cannot be null.

# _drill-down data - inverse of roll-up
