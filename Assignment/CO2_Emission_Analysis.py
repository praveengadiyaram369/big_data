"""

Analyzing country's CO2 emissions data using Apache Spark. 

Source Code Authors:

1) Naga Kartheek Reddy Kona : 219203205
2) Kausik Kappaganthula : 219203215
3) Sowjanya Chennamaneni:219203249
4) Sri Sai Praveen Gadiyaram : 219203192
5) Arun Anirudhan  : 219203285

"""

# _importing pyspark libraries
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

# _importing libraries needed for visualization
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import numpy as np


def get_change_in_percentage(a, b):
    return ((b - a) * 100)/a    # _absolute change in data


def get_normalized_value(val, mean_change, std_change):
    return (val - mean_change)/std_change       # _normalized z-score


def load_data(spark, file_name):

    # _load csv file to dataframe
    data_df = spark.read.option('inferSchema', 'true').option(
        'header', 'true').csv(file_name)

    return data_df


def show_scatter_plot(emission_data):

    country_marker_list = []
    x_values = []
    y_values = []

    # _preparing data for scatter plot representing the clustering result
    for row in emission_data:
        country_marker_list.append(int(row["Incomelevel"]))
        x_values.append(float(row["change_in_emissions_scaled"]))
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

    # _create legend for each income group
    plt.legend((li, lmi, umi, hi),
               ('Low income', 'Lower middle income',
                'Upper middle income', 'High income'),
               scatterpoints=1,
               loc='best',
               fontsize=8)

    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.title("CO2 emissions Clustering k=7 representing country's Incomegroup ")
    plt.ylabel("Cluster Id")
    plt.xlabel("Percentage change(normalized) in CO2 emissions from 2004 to 2014")

    # _saving plot result to an image
    plt.savefig('results/plots/' +
                'cluster_result_with_income.png')
    plt.show()


def show_country_wise_clustering(co2_emisssion_data):

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
    plot_data_df.plot(column='prediction',
                      ax=ax, legend=True)

    ax.set_title("Countries clustered by K-means(k=7)", fontsize=12)
    # saving the plot image in result folder
    plt.savefig('results/plots/' +
                'countries_clustered_by_co2_emissions.png', bbox_inches='tight')
    plt.show()


def perform_data_preprocessing(spark):

    # _load dataframes
    co2_emisssion_data = load_data(spark, "data/CO2E_data.csv")
    country_meta_data = load_data(spark, "data/Metadata_Country_CO2E_data.csv")
    income_df = spark.createDataFrame(
        [("Low income", 40), ("Lower middle income", 30),
         ("Upper middle income", 20), ("High income", 10)],
        ["IncomeGroup", "Incomelevel"])

    # _data pre-processing -- select only columns needed for analysis
    co2_emisssion_data = co2_emisssion_data.select(
        "Country Name", "Country Code", "2004", "2014").dropDuplicates()  # _drop duplicate rows

    # _filter the rows which have no values for all 2004, 2014
    co2_emisssion_data = co2_emisssion_data.na.drop(
        "any", subset=("2004", "2014"))

    # adding the column change_in_emissions contains changes of co2 emissions from 2004 to 2014
    co2_emisssion_data = co2_emisssion_data.withColumn(
        'change_in_emissions', get_change_in_percentage(co2_emisssion_data['2004'], co2_emisssion_data['2014']))
    co2_emisssion_data = co2_emisssion_data.orderBy("change_in_emissions")

    mean_change, std_change = co2_emisssion_data.select(
        mean("change_in_emissions"), stddev("change_in_emissions")).first()
    co2_emisssion_data = co2_emisssion_data.withColumn(
        'change_in_emissions_scaled', get_normalized_value(co2_emisssion_data['change_in_emissions'], mean_change, std_change))
    # removing outliers that are 3 standard deviations away from the mean
    co2_emisssion_data = co2_emisssion_data.where(
        col('change_in_emissions_scaled') < 3)

    # _load country meta_data income levels
    country_meta_data = country_meta_data.select("Country Code", "IncomeGroup")
    country_meta_data = country_meta_data.join(
        income_df, ["IncomeGroup"])

    # _performing natural join to map emission data with country's income group
    co2_emisssion_data = co2_emisssion_data.join(
        country_meta_data, ["Country Code"])

    co2_emisssion_data = co2_emisssion_data.na.drop(
        subset=("IncomeGroup"))  # _filter null IncomeGroup rows

    return co2_emisssion_data


def analysing_emissions_data(spark, co2_emisssion_data):

    # creating feature vector for sending as input to ML models
    vecAssembler = VectorAssembler(
        inputCols=['change_in_emissions_scaled'], outputCol="features")

    # adding feature vector to our aperk dataframe
    co2_emisssion_data = vecAssembler.setHandleInvalid(
        "skip").transform(co2_emisssion_data)

    # creating Kmeans object (7 clusters)
    kmeans = KMeans(k=7)

    # clustering operation
    model = kmeans.fit(co2_emisssion_data.select('features'))

    # adding column of predicted clusters to our dataframe
    co2_emisssion_data = model.transform(co2_emisssion_data)

    return co2_emisssion_data.drop("features")


def plot_clustering_result(co2_emisssion_data):

    show_scatter_plot(co2_emisssion_data.collect())
    show_country_wise_clustering(co2_emisssion_data)


def perform_correlation_analysis(co2_emisssion_data):

    # _look at average emissions for each income group
    co2_emisssion_data.show(5)
    co2_emisssion_data.select("IncomeGroup", "change_in_emissions_scaled").groupBy(
        "IncomeGroup").agg(avg("change_in_emissions_scaled").alias('average_change_in_emissions')).orderBy("average_change_in_emissions").show()

    # _perform correlation on country's decade change in emissions and income level
    corr_decade_change_income = co2_emisssion_data.stat.corr(
        "change_in_emissions", "Incomelevel")

    print(
        f"Correlation between country's 2004-2014 decade co_2 emission changes and income levels: {corr_decade_change_income}")


if __name__ == "__main__":

    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _start spark sessin from context
    spark = SparkSession(sc)

    # _load data and perform analysis
    co2_emisssion_data = perform_data_preprocessing(spark)

    # _analysing data using KMeans
    co2_emisssion_data = analysing_emissions_data(spark, co2_emisssion_data)

    # _plotting co2 emissions in geopandas
    plot_clustering_result(co2_emisssion_data)

    # _performing correlation between decade emissions and countrys income level
    perform_correlation_analysis(co2_emisssion_data)

    # _stop spark context and end the process
    sc.stop()
