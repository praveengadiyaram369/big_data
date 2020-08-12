from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import OneHotEncoder

if __name__ == "__main__":

    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _start spark sessin from context
    spark = SparkSession(sc)

    df = spark.createDataFrame([
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (0.0, 1.0),
        (2.0, 0.0)
    ], ["categoryIndex1", "categoryIndex2"])

    encoder = OneHotEncoder(inputCols=["categoryIndex1", "categoryIndex2"],
                            outputCols=["categoryVec1", "categoryVec2"])
    model = encoder.fit(df)
    encoded = model.transform(df)
    encoded.show()

    sc.stop()
