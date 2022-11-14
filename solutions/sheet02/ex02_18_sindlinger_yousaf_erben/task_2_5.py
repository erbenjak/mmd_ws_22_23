import numpy as np
import pyspark
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml.stat import Correlation
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import expr, udf, col, countDistinct, dense_rank
from pyspark.sql.types import LongType, StringType, StructField, StructType, BooleanType, ArrayType, IntegerType, \
    FloatType


def create_spark_session():
    spark = SparkSession.builder.appName('Recommendations').getOrCreate()
    return spark


def custom_replace(artistid, badid, goodid):
    if (artistid == badid):
        return goodid
    else:
        return artistid


def merge_two_dicts(x, y):
    z = x.copy()  # start with keys and values of x
    z.update(y)  # modifies z with keys and values of y
    return z


def create_utility_matrix(spark):
    # this has to follow a simple three-step plan
    # 1. load the user data
    user_schema = StructType([
        StructField("userid", IntegerType(), True),
        StructField("artistid", IntegerType(), True),
        StructField("playcount", FloatType(), True)])

    user_ratings = spark.read.csv("data_sheet2/user_artist_data_small.txt", header="false", sep=' ',
                                  schema=user_schema)
    # ------- debug checks for correct loading ------------
    # user_ratings.printSchema()
    # user_ratings.show(10)

    # 2. load the artist aliases database
    artist_aliases_schema = StructType([
        StructField("badid", LongType(), True),
        StructField("goodid", LongType(), True)])

    artist_aliases = spark.read.csv("data_sheet2/artist_alias_small.txt", header="false", sep='\t',
                                    schema=artist_aliases_schema)
    # ------- debug checks for correct loading ------------
    # artist_aliases.printSchema()
    # artist_aliases.show(10)

    # replace the bad with the good ids
    # firstly we outer join
    user_ratings_corrected = user_ratings.join(artist_aliases, user_ratings.artistid == artist_aliases.badid, "left")
    # secondly we recombine the colums
    conversionUDF = udf(lambda x, y, z: custom_replace(x, y, z), IntegerType())
    user_ratings_corrected = user_ratings_corrected.withColumn("artistid_corrected",
                                                               conversionUDF(col("artistid"), col("badid"),
                                                                             col("goodid")))
    user_ratings_corrected = user_ratings_corrected.select(col("userid"), col("artistid_corrected").alias("artistid"),
                                                           col("playcount")).orderBy("userid")
    # user_ratings_corrected.show(5)
    # user_ratings_corrected.printSchema()

    # 3. create a sparse matrix - this will lower the storage and processing effort associated with the
    # utility matrix
    matrix = CoordinateMatrix(user_ratings_corrected.rdd)
    return matrix


def extract_dense_user_vector(utility_mat, user_id):
    result = utility_mat.entries
    # we need to compute the maximal artist id
    max_artist_id = result.max(lambda x: x.j).j

    # now we need to select all entries with the fitting user id
    results_filtered = result.filter(lambda x: x.i == user_id)
    sparse_vector_as_dict = results_filtered.map(lambda x: {x.j: x.value}).reduce(lambda x, y: merge_two_dicts(x, y))

    sparse_vector = SparseVector(max_artist_id + 1, sparse_vector_as_dict)
    return sparse_vector.toArray()


def get_unique_user_ids(utility_mat):
    result = utility_mat.entries
    return result.map(lambda x: x.i).distinct().collect()


def calculate_pearson_corr_matrix(user1_vector, user2_vector, spark):
    user1 = np.array(user1_vector)
    user2 = np.array(user2_vector)
    return np.corrcoef(user1, user2)[0][1]


def recommender_system_1():
    # task a)
    spark = create_spark_session()
    utility_matrix = create_utility_matrix(spark)
    print("TASK A)")
    print("The utility matric is storred in a coordinate Matrix: " + str(utility_matrix) + "\n")

    # task b)
    print("TASK B)")
    print("We try to determine the coefficient between two random employees")
    user1_vector = extract_dense_user_vector(utility_matrix, 2010008)
    print("Note: Since we could not get the pyspark coefficient function to work we convert the "
          "vector numpy-arrays and use its pearson correlation function")
    user2_vector = extract_dense_user_vector(utility_matrix, 1041919)
    pearsonCorr = calculate_pearson_corr_matrix(user1_vector, user2_vector, spark)
    print('The peason coefficient between user 2010008 und 1041919 is: ' + str(pearsonCorr))
    print("\n")

    # task c)
    print("TASK C) - a all correlation coefficients are determined and than sorted")
    unique_ids = get_unique_user_ids(utility_matrix)

    vector_dict = {}
    j = 0
    for i in unique_ids:
        vector_dict.update({i: extract_dense_user_vector(utility_matrix, i)})
        j += 1

    correlation_dic = []
    adder = 1
    for i in unique_ids:
        for j in unique_ids:
            if i == j:
                continue
            if i > j:
                continue

            user1_vector = vector_dict[i]
            user2_vector = vector_dict[j]
            corr = calculate_pearson_corr_matrix(user1_vector, user2_vector, spark)
            correlation_dic.append([i, j, float(corr)])

        adder += 1

    result_schema = StructType([
        StructField("userid1", IntegerType(), True),
        StructField("userid2", IntegerType(), True),
        StructField("correlation", FloatType(), True)])
    correlation_results = spark.sparkContext.parallelize(correlation_dic).toDF(result_schema)

    # please note that this far - due to the slowness only a few user accounts are analyzed
    user_to_search_for = 1000647
    k = 5
    print("showing the top 5 results for user 1000647")
    correlation_results = correlation_results.select(col("userid1"), col("userid2"), col("correlation")).where(
        (col("userid1") == user_to_search_for) | (col("userid2") == user_to_search_for)).orderBy(col("correlation").desc())
    correlation_results.show(k)
    print("\n")


if __name__ == '__main__':
    recommender_system_1()
