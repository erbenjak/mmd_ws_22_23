from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, IntegerType, FloatType
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.evaluation import RegressionMetrics




def create_spark_session():
    spark = SparkSession.builder.appName('Recommendations').getOrCreate()
    return spark


def load_data_into_ratings(spark, sd):
    # load the movie lens data
    r_schema = StructType([
        StructField("userid", IntegerType(), True),
        StructField("itemid", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", IntegerType(), True),
    ])
    ratings = spark.read.csv("data_sheet3/movielens.txt", header="false", sep='\t',
                                  schema=r_schema)

    # map each row to ratings 
    ratings_mapped = ratings.rdd.map(lambda row: Rating(row[0], row[1], row[2]))
    (ratings_mapped_train, ratings_mapped_test) = ratings_mapped.randomSplit([0.5, 0.5], seed=sd)
    return ratings_mapped_train, ratings_mapped_test


def als_builder(training_data, sd):

    # latent factors
    rank = 10
    # iterations
    numIterations = 5
 
    model = ALS.train(ratings=training_data, rank=rank, iterations=numIterations, seed=sd)
    return model

def als_prediction(model:MatrixFactorizationModel, test_data):
    prediction = model.predictAll(test_data.map(lambda row: (row[0], row[1])))
    return prediction

def calculate_mse(test_data, prediction_data):
    # preprocess data for joining: rdd joins need to be in (key, value) type when joining
    test_data_modified = test_data.map(lambda row:((row.user, row.product), row.rating))
    prediction_data_modified = prediction_data.map(lambda row:((row.user, row.product), row.rating))
    
    # join data
    scoreAndLabels = prediction_data_modified.join(test_data_modified).map(lambda tup: tup[1])

    # print(scoreAndLabels.take(10))

    # instantiate regression metrics to compare predicted and actual ratings
    metrics = RegressionMetrics(scoreAndLabels)

    # return mse
    return metrics.meanSquaredError




def recommender_system_als():

    seed = 1234

    spark = create_spark_session()

    # part a)
    # load data, generate training and test sample
    ratings_mapped_train, ratings_mapped_test = load_data_into_ratings(spark, seed)

    # part b)
    # build als model
    als_model = als_builder(ratings_mapped_train, seed)

    # save model
    als_model.save(spark.sparkContext, "data_sheet3/task_3_3_als.model")

    # part c)
    # prediction
    prediction_data = als_prediction(als_model, ratings_mapped_test)

    # calculate mse
    mse = calculate_mse(ratings_mapped_test, prediction_data)

    print("Result using seed %s:" % seed)
    print("*** MSE = %s ***" % mse)
    # 1.4155900724882908

if __name__ == '__main__':
    recommender_system_als()
