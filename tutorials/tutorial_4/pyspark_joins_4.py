from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

if __name__ == "__main__":
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

    # _pivot the data
    retail_pivot = retail_data.groupBy(
        "StockCode").pivot("Country").sum("Quantity").orderBy("StockCode")
    retail_pivot = retail_pivot.na.fill(0)
    retail_pivot.show(5)

    customers = spark.read.option('inferSchema', 'true').option(
        'header', 'true').csv('customers.csv')
    orders = spark.read.option('inferSchema', 'true').option(
        'header', 'true').csv('orders.csv')
    orders2 = spark.read.option('inferSchema', 'true').option(
        'header', 'true').csv('orders2.csv')

    customers.show()
    orders.show()
    orders2.show()

    # _natural join
    join_result = customers.join(orders, ["customer_id"])
    join_result.show()

    # _inner join
    join_expression = customers["customer_id"] == orders["customer_id"]
    join_type = "inner"
    join_result = customers.join(orders, join_expression, join_type)
    join_result.show()

    # _outer join
    join_type = "outer"
    join_result = customers.join(orders, join_expression, join_type)
    join_result.show()

    # _left outer join
    join_type = "left_outer"
    join_result = customers.join(orders, join_expression, join_type)
    join_result.show()

    # _right outer join
    join_type = "right_outer"
    join_result = customers.join(orders, join_expression, join_type)
    join_result.show()

    # _semi join
    join_type = "left_semi"
    join_result = customers.join(orders, join_expression, join_type)
    join_result.show()

    # _anti join
    join_type = "left_anti"
    join_result = customers.join(orders, join_expression, join_type)
    join_result.show()

    # _cross join
    join_result = customers.crossJoin(orders)
    join_result.show()

    spark.stop()
