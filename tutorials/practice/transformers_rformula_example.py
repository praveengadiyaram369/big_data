from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import RFormula

if __name__ == "__main__":

    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _start spark sessin from context
    spark = SparkSession(sc)

    dataset = spark.createDataFrame(
        [(7, "US", 18, 1.0),
        (8, "CA", 12, 0.0),
        (9, "NZ", 15, 0.0)],
        ["id", "country", "hour", "clicked"])

    formula = RFormula(
        formula="clicked ~ country + hour",
        featuresCol="features",
        labelCol="label")

    output = formula.fit(dataset).transform(dataset)
    output.select("features", "label").show()

    sc.stop()

