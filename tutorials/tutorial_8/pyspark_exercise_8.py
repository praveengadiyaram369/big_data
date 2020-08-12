from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import LogisticRegression

if __name__ == "__main__":

    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _start spark sessin from context
    spark = SparkSession(sc)

    # _import zoo data to a spark dataframe
    zoo_df = spark.read.option("inferschema", "true").option(
        "header", "true").csv("zoo.csv")
    zoo_df.show(5)
    zoo_df.printSchema()

    # _add new column Is_Mammal
    zoo_df = zoo_df.withColumn("Is_Mammal", expr(
        "CASE WHEN Type = 1 THEN 1 ELSE 0 END"))

    # _preprocess data
    pre_process_data = RFormula(
        formula="Is_Mammal ~ Hair + Feathers + Eggs + Milk + Airborne + Aquatic + Predator + Toothed + Backbone + Breathes + Venomous + Fins + Legs + Tail + Domestic + Catsize")
    pre_process_data = pre_process_data.fit(zoo_df)
    pre_process_data = pre_process_data.transform(zoo_df)

    pre_process_data.show(5)

    # _split dataset into test and train datasets
    train, test = pre_process_data.randomSplit([0.7, 0.3])

    # _initialize logistic regression classifier
    lr = LogisticRegression(labelCol="label", featuresCol="features")

    # _train logistic regression model with train data available
    fittedLr = lr.fit(train)

    # _classify test data
    result = fittedLr.transform(test)
    result.show()

    # _compare mammal type column against predictions
    result.select("AnimalName", "Is_Mammal", "prediction").show()

    spark.stop()
