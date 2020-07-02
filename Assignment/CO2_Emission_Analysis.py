# _importing pyspark libraries
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StringIndexer

# _importing libraries needed for visualization
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import numpy as np


def get_change_in_percentage(a, b):
    return ((b - a) * 100)/a


def get_country_income_indexer():

    income_df = spark.createDataFrame(
        [("Low income", 4), ("Lower middle income", 3),
         ("Upper middle income", 2), ("High income", 1)],
        ["IncomeGroup", "income_level"])

    indexer = StringIndexer(inputCol="IncomeGroup", outputCol="Incomelevel")
    return indexer


def save_clustering_result(emission_data):

    country_marker_list = []
    x_values = []
    y_values = []

    for row in emission_data:
        country_marker_list.append(int(row["Incomelevel"]))
        x_values.append(float(row["change_in_emissions"]))
        y_values.append(int(row["prediction"]))

    country_marker_array = np.array(country_marker_list)
    x_array = np.array(x_values)
    y_array = np.array(y_values)

    fig, ax = plt.subplots()

    hi = ax.scatter(x_array[country_marker_array == 10],
                    y_array[country_marker_array == 10], marker='*')
    umi = ax.scatter(x_array[country_marker_array == 20],
                     y_array[country_marker_array == 20], marker='x')
    lmi = ax.scatter(x_array[country_marker_array == 30],
                     y_array[country_marker_array == 30], marker='s')
    li = ax.scatter(x_array[country_marker_array == 40],
                    y_array[country_marker_array == 40], marker='^')

    plt.legend((li, lmi, umi, hi),
               ('Low income', 'Lower middle income',
                'Upper middle income', 'High income'),
               scatterpoints=1,
               loc='upper right',
               fontsize=8)

    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.title("CO2 emissions Clustering k=5 representing country's Incomegroup ")
    plt.ylabel("Cluster Id")
    plt.xlabel("Percentage increase of CO2 emissions from 2004 to 2014")

    plt.savefig('results/plots/' +
                'cluster_result_with_income.png')
    plt.show()


def load_data(spark, file_name):

    # _load csv file to dataframe
    data_df = spark.read.option('inferSchema', 'true').option(
        'header', 'true').csv(file_name)

    return data_df


def perform_data_preprocessing(spark):

    # _load dataframes
    co2_emisssion_data = load_data(spark, "data/CO2E_data.csv")
    country_meta_data = load_data(spark, "data/Metadata_Country_CO2E_data.csv")

    # _data pre-processing -- select only columns needed for analysis
    emission_data_df = co2_emisssion_data.select(
        "Country Name", "Country Code", "2004", "2014")

    # adding the column change_in_emissions contains changes of co2 emissions from 2004 to 2014
    emission_data_df = emission_data_df.withColumn(
        'change_in_emissions', get_change_in_percentage(emission_data_df['2004'], emission_data_df['2014']))

    emission_data_df = emission_data_df.dropDuplicates()  # _drop duplicate rows

    # _filter the rows which have no values for all 1994, 2004, 2014
    emission_data_df = emission_data_df.na.drop(
        "any", subset=("2004", "2014"))

    income_df = spark.createDataFrame(
        [("Low income", 40), ("Lower middle income", 30),
         ("Upper middle income", 20), ("High income", 10)],
        ["IncomeGroup", "Incomelevel"])

    # _load country meta_data income levels
    country_meta_data = country_meta_data.select("Country Code", "IncomeGroup")
    country_meta_data = country_meta_data.join(
        income_df, ["IncomeGroup"])

    # _performing natural join to map emission data with country's income group
    emission_data_df = emission_data_df.join(
        country_meta_data, ["Country Code"])

    emission_data_df = emission_data_df.na.drop(subset=("IncomeGroup"))

    emission_data_df.show(10)

    return emission_data_df


def analysing_emissions_data(spark, co2_emisssion_data):

    # creating feature vector for sending as input to ML models
    vecAssembler = VectorAssembler(
        inputCols=['change_in_emissions'], outputCol="features")

    # adding feature vector to our aperk dataframe
    co2_emisssion_data = vecAssembler.setHandleInvalid(
        "skip").transform(co2_emisssion_data)

    # creating Kmeans object (5 clusters)
    kmeans = KMeans(k=5)

    # clustering operation
    model = kmeans.fit(co2_emisssion_data.select('features'))

    # adding column of predicted clusters to our dataframe
    co2_emisssion_data = model.transform(co2_emisssion_data)

    save_clustering_result(co2_emisssion_data.collect())

    return co2_emisssion_data.drop("features")


def plot_co2_emissions(co2_emisssion_data):

    # reading each country geometry, iso code etc into data frame
    world = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres'))

    # getting geometry column and iso_a3 (3 letter code of country)
    country_shapes = world[['geometry', 'iso_a3']]

    # coverting spark dataframe to pandas
    co2_emisssion_data_df = co2_emisssion_data.toPandas()

    # merging country_shapes, co2_emisssion_data_df using country code and converting into geodataframe
    plot_data_df = geopandas.GeoDataFrame(pd.merge(
        co2_emisssion_data_df, country_shapes, left_on='Country Code', right_on='iso_a3'))

    fig, ax = plt.subplots(1, 1)

    # plotting based on clusters that country belongs to
    plot_data_df.plot(column='prediction', ax=ax, legend=True)

    plt.savefig('results/plots/' +
                'countries_clustered_by_co2_emissions.png', bbox_inches='tight')
    plt.show()


def perform_correlation_analysis(co2_emisssion_data):

    co2_emisssion_data.show(5)
    corr_2014_income = co2_emisssion_data.stat.corr("2014", "Incomelevel")
    corr_decade_change_income = co2_emisssion_data.stat.corr(
        "change_in_emissions", "Incomelevel")

    print(
        f"Correlation between country's 2014 co_2 emissions and income levels -- {corr_2014_income}")
    print(
        f"Correlation between country's 2004-2014 decade co_2 emission changes and income levels -- {corr_decade_change_income}")


if __name__ == "__main__":

    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _start spark sessin from context
    spark = SparkSession(sc)

    # _load data and perform analysis
    emission_data_df = perform_data_preprocessing(spark)

    # analysing data using KMeans
    co2_emisssion_data = analysing_emissions_data(spark, emission_data_df)

    # plotting co2 emissions in geopandas
    plot_co2_emissions(co2_emisssion_data)

    perform_correlation_analysis(co2_emisssion_data)

    # _stop spark context and end the process
    sc.stop()
