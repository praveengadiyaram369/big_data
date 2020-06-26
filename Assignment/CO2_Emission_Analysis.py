from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *


def load_data(spark, file_name):

    # _load csv file to dataframe
    data_df = spark.read.option('inferSchema', 'true').option(
        'header', 'true').csv(file_name)

    return data_df


def perform_emission_analysis(spark):

    # _load dataframes
    co2_emisssion_data = load_data(spark, "data/CO2E_data.csv")
    country_meta_data = load_data(spark, "data/Metadata_Country_CO2E_data.csv")

    # _data pre-processing
    # _select only columns needed for analysis
    emission_data_df = co2_emisssion_data.select(
        "Country Name", "Country Code", "1994", "2004", "2014")
    country_meta_data = country_meta_data.select("Country Code", "IncomeGroup")

    # _performing natural join to map emission data with country's income group
    emission_data_df = emission_data_df.join(
        country_meta_data, ["Country Code"])
    emission_data_df = emission_data_df.dropDuplicates().orderBy(
        "Country Code")  # _drop duplicate rows

    # _replacing empty "IncomeGroup" with the value "NOT_A_COUNTRY"
    emission_data_df = emission_data_df.na.fill(
        {"IncomeGroup": "NOT_A_COUNTRY"})

    # _filter the rows which have no values for all 1994, 2004, 2014
    emission_data_df = emission_data_df.na.drop(
        "all", subset=("1994", "2004", "2014"))

    emission_data_df.show(10)

    # _write the processed dataframe data to a csv file
    emission_data_df.coalesce(1).write.format('csv').option(
        'header', 'true').save('results/emission_data_df')


if __name__ == "__main__":

    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _start spark sessin from context
    spark = SparkSession(sc)

    # _load data and perform analysis
    perform_emission_analysis(spark)

    # _stop spark context and end the process
    sc.stop()
