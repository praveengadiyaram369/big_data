from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, col

# _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.

sc = SparkContext("local", "Hello World")
sc.setLogLevel('ERROR')

# _start spark sessin from context
spark = SparkSession(sc)
range = spark.range(100).toDF('numbers')
range.show(5)

# _select only the values which are divisible by 3
div_by_3 = range.where("numbers % 3 == 0")
div_by_3.show(5)

# _read csv file
flightdata = spark.read.option('inferSchema', 'true').option(
    'header', 'true').csv('2015-summary.csv')
flightdata.show(5)
flightdata.printSchema()

# _sorted data descending order
sorted_flightdata = flightdata.sort(desc('count'))
sorted_flightdata.show(5)

# _filter data using sql
flightdata.createOrReplaceTempView('flightTable')
spark.sql('''SELECT * from flightTable where count > 10 ''').show()

# _filter data using col
flightdata.where(col('count') > 10).show()
flightdata.filter(col('count') > 10).show()

flights_US = flightdata.filter(col('DEST_COUNTRY_NAME') == 'United States')
flights_US.show(5)

flights_US_India = flightdata.filter(
    (col('DEST_COUNTRY_NAME') == 'United States') & (col('ORIGIN_COUNTRY_NAME') == 'India'))
flights_US_India.show(5)

# _RDD usage
rdd = sc.textFile('alice.txt')
alice_counts = rdd.flatMap(lambda line: line.split(' ')).map(
    lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
# _creates folder and includes partitions inside
alice_counts.saveAsTextFile('Alice_counts')
