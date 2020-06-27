from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import geopandas
import matplotlib.pyplot as plt


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


def analysing_emissions_data(spark):

    # load dataframe
    co2_emisssion_data = load_data(spark, "results/emission_data_df/part-00000-7b469c18-67d3-44ce-9c02-094bfc25559d-c000.csv")

    # adding the column change_in_emissions contains changes of co2 emissions from 2004 to 2014
    co2_emisssion_data = co2_emisssion_data.withColumn('change_in_emissions', co2_emisssion_data['2014'] - co2_emisssion_data['2004'])

    # creating feature vector for sending as input to ML models
    vecAssembler = VectorAssembler(inputCols=['change_in_emissions'], outputCol="features")

    # adding feature vector to our aperk dataframe
    co2_emisssion_data = vecAssembler.setHandleInvalid("skip").transform(co2_emisssion_data)

    # creating Kmeans object (7 clusters)
    kmeans = KMeans(k=7)

    # clustering operation
    model = kmeans.fit(co2_emisssion_data.select('features'))

    # adding column of predicted clusters to our dataframe
    co2_emisssion_data = model.transform(co2_emisssion_data)

    co2_emisssion_data.show(10)

    return co2_emisssion_data


def plot_co2_emissions(co2_emisssion_data):

    # reading each country geometry, iso code etc into data frame
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    # getting geometry column and iso_a3 (3 letter code of country)
    country_shapes = world[['geometry', 'iso_a3']]
    
    # coverting spark dataframe to pandas
    co2_emisssion_data_df = co2_emisssion_data.toPandas()

    # merging country_shapes, co2_emisssion_data_df using country code and converting into geodataframe
    plot_data_df = geopandas.GeoDataFrame(pd.merge(co2_emisssion_data_df, country_shapes, left_on='Country Code', right_on='iso_a3'))
    
    fig, ax = plt.subplots(1, 1)
    
    # plotting based on clusters that country belongs to
    plot_data_df.plot(column='prediction', ax=ax, legend=True)
    plt.show()

if __name__ == "__main__":

    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _start spark sessin from context
    spark = SparkSession(sc) 

    # _load data and perform analysis
    perform_emission_analysis(spark)

    # analysing data using KMeans
    co2_emisssion_data = analysing_emissions_data(spark)

    # plotting co2 emissions in geopandas
    plot_co2_emissions(co2_emisssion_data)

    # _stop spark context and end the process
    sc.stop()
