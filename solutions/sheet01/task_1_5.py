from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import when, col, mean, desc, round, translate
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, IntegerType, StringType


def init_spark():
    spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
    sc = spark.sparkContext
    return spark, sc

def init_adult_dataframe(spark, sc, name):
    # first we load the data into a rdd as proposed by the exercise sheet
    rdd_with_data = sc.textFile(name)
    rdd_split_by_delimiter = rdd_with_data.map(lambda l: l.split(","))

    # Convert to Row
    rdd_as_ordered_struct = rdd_split_by_delimiter.map(
        lambda p: Row(age=p[0], workclass=p[1], fnlwgt=p[2], education=p[3], educationNumber=p[4], maritalStatus=p[5],
                      occupation=p[6], relationship=p[7], race=p[8], sex=p[9], capitalGain=p[10], capitalLoss=p[11],
                      hoursPerWeek=p[12], nativeCountry=p[13], income=p[14]))

    # now we can go ahead and create the dataframe
    # dataframe_adults = rdd_with_data.toDF(rdd_as_ordered_struct)
    dataframe_adults = spark.createDataFrame(rdd_as_ordered_struct)

    # we need to fix some datatypes
    dataframe_adults = dataframe_adults.withColumn("age", col("age").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("fnlwgt", col("fnlwgt").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("educationNumber", col("educationNumber").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("capitalGain", col("capitalGain").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("capitalLoss", col("capitalLoss").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("hoursPerWeek", col("hoursPerWeek").cast("double"))

    # now we will check if the dataframe is in order quickly
    # dataframe_adults.printSchema()
    # dataframe_adults.show(20)
    return dataframe_adults


def main():
    # initialize spark contexts and creating the dataframe necessary
    spark, sc = init_spark()
    train_df = init_adult_dataframe(spark, sc, "adult.train")
    test_df = init_adult_dataframe(spark, sc, "adult.test")

    categorical_variables = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
                             'nativeCountry']
    indexers = [StringIndexer(inputCol=column, outputCol=column + "-index") for column in categorical_variables]
    encoder = OneHotEncoder(
        inputCols=[indexer.getOutputCol() for indexer in indexers],
        outputCols=["{0}-encoded".format(indexer.getOutputCol()) for indexer in indexers]
    )
    assembler = VectorAssembler(
        inputCols=encoder.getOutputCols(),
        outputCol="categorical-features"
    )
    pipeline = Pipeline(stages=indexers + [encoder, assembler])
    train_df = pipeline.fit(train_df).transform(train_df)
    test_df = pipeline.fit(test_df).transform(test_df)

    continuous_variables = ['age', 'fnlwgt', 'educationNumber', 'capitalGain', 'capitalLoss', 'hoursPerWeek']

    assembler = VectorAssembler(
        inputCols=['categorical-features', *continuous_variables],
        outputCol='features'
    )
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    indexer = StringIndexer(inputCol='income', outputCol='label')
    train_df = indexer.fit(train_df).transform(train_df)
    test_df = indexer.fit(test_df).transform(test_df)

    # train_df.limit(10).toPandas()['label']
    # test_df.limit(10).toPandas()['label']

    lr = LogisticRegression(featuresCol='features', labelCol='label')
    model = lr.fit(train_df)

    print(model.summary)

    pred = model.transform(test_df)
    print(pred.limit(10).toPandas()[['label', 'prediction']])


if __name__ == '__main__':
    main()
