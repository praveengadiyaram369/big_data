import re
import string

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, col

regex = re.compile('[%s]' % re.escape(string.punctuation))


def remove_punctuation(s):
    return regex.sub('', s)


def get_proper_words(line):

    word_list = [word.lower() for word in remove_punctuation(
        line).split(' ')]
    return word_list


if __name__ == "__main__":

    # _we can use spark in either local mode or cluster mode. Below is the configuration for local mode.
    sc = SparkContext("local", "Hello World")
    sc.setLogLevel('ERROR')

    # _RDD usage
    rdd = sc.textFile('alice.txt')
    word_list = rdd.flatMap(lambda line: get_proper_words(line))
    word_list = word_list.filter(lambda word: len(word) > 0)
    word_list = word_list.map(
        lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
    word_list = word_list.coalesce(1)
    word_list = word_list.sortByKey(True)

    # _save data as a folder and include partitions inside
    word_list.saveAsTextFile('Alice_counts')

    sc.stop()
