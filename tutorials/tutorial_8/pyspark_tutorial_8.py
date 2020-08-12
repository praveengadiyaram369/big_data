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
    mushroom_df = spark.read.option("inferschema", "true").option(
        "header", "true").csv("mushrooms.csv")
    mushroom_df.show(5)
    mushroom_df.printSchema()

    mushroom_df = mushroom_df.na.drop()
    # _No need to create extra column as Lab column is already binary classifiable with either EDIBLE or POISONOUS values
    mushroom_df = mushroom_df.drop("VeilType")

    # _preprocess data
    pre_process_data = RFormula(
        formula="Lab ~ .")
    pre_process_data = pre_process_data.fit(mushroom_df)
    pre_process_data = pre_process_data.transform(mushroom_df)

    pre_process_data.show(5)

    # _split dataset into test and train datasets
    train, test = pre_process_data.randomSplit([0.7, 0.3])

    # _initialize logistic regression classifier
    lr = LogisticRegression(labelCol="label", featuresCol="features")

    # _train logistic regression model with train data available
    fittedLr = lr.fit(train)

    # _classify test data
    result = fittedLr.transform(test)
    result.show(5)

    # _compare mammal type column against predictions
    result.select("label", "prediction").show(5)

    true_posiive = result.filter(
        expr("label = 1.0 and prediction = 1.0")).count()
    false_positive = result.filter(
        expr("label = 0.0 and prediction = 1.0")).count()

    true_negative = result.filter(
        expr("label = 0.0 and prediction = 0.0")).count()
    false_negative = result.filter(
        expr("label = 1.0 and prediction = 0.0")).count()

    print(f"True Positive: {true_posiive} \n False Positive: {false_positive} \n True Negative: {true_negative} \n False Negative: {false_negative} \n")

    recall = (true_posiive/(true_posiive + false_negative))
    precision = (true_posiive/(true_posiive + false_positive))
    accuracy = ((true_posiive + true_negative)/(true_posiive +
                                                true_negative + false_positive + false_negative))

    print(
        f"Accuracy: {accuracy} \n Recall: {recall} \n Precision: {precision}")

    spark.stop()
