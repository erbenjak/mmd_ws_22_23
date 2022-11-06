import pyspark
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import expr, udf, col, countDistinct, dense_rank
from pyspark.sql.types import LongType, StringType, StructField, StructType, BooleanType, ArrayType, IntegerType


def create_spark_session():
    spark = SparkSession.builder.appName('Recommendations').getOrCreate()
    return spark


def custom_replace(artistid, badid, goodid):
    if (artistid == badid):
        return goodid
    else:
        return artistid


def create_utility_matrix(spark):
    # this has to follow a simple three-step plan
    # 1. load the user data
    user_schema = StructType([
        StructField("userid", IntegerType(), True),
        StructField("artistid", IntegerType(), True),
        StructField("playcount", IntegerType(), True)])

    user_ratings = spark.read.csv("data_sheet2/user_artist_data_small.txt", header="false", sep=' ',
                                  schema=user_schema)
    # ------- debug checks for correct loading ------------
    # user_ratings.printSchema()
    # user_ratings.show(10)

    # 2. load the artist aliases database
    artist_aliases_schema = StructType([
        StructField("badid", IntegerType(), True),
        StructField("goodid", IntegerType(), True)])

    artist_aliases = spark.read.csv("data_sheet2/artist_alias_small.txt", header="false", sep='\t',
                                    schema=artist_aliases_schema)
    # ------- debug checks for correct loading ------------
    # artist_aliases.printSchema()
    # artist_aliases.show(10)

    # replace the bad with the good ids
    # firstly we outer join

    unique_users = user_ratings.select(col("userid")).distinct().withColumn("new_user_id", dense_rank().over(
                                                                                Window.orderBy('userid')))

    user_ratings_corrected = user_ratings.join(artist_aliases, user_ratings.artistid == artist_aliases.badid, "left")
    user_ratings_corrected = user_ratings_corrected.join(unique_users, user_ratings_corrected.userid == unique_users.userid,"left")
    #user_ratings_corrected.show(500)
    # secondly we recombine the colums
    conversionUDF = udf(lambda x, y, z: custom_replace(x, y, z), IntegerType())
    user_ratings_corrected = user_ratings_corrected.withColumn("artistid_corrected",
                                                               conversionUDF(col("artistid"), col("badid"),
                                                                             col("goodid")))
    user_ratings_corrected = user_ratings_corrected.select(col("new_user_id").alias("userid"), col("artistid_corrected").alias("artistid"),
                                                           col("playcount")).orderBy("userid")
    user_ratings_corrected.show(10000)

    # 3. create a sparse matrix - this will lower the storrage and processing effort asociated with the
    # utility matrix
    matrix = CoordinateMatrix(user_ratings_corrected.rdd.map(lambda coords: MatrixEntry(*coords)))
    return matrix.transpose().toRowMatrix()


def recommender_system_1():
    spark = create_spark_session()
    utility_matrix = create_utility_matrix(spark)
    print(utility_matrix.numRows())
    print(utility_matrix.numCols())


if __name__ == '__main__':
    recommender_system_1()
