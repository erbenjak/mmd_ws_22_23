from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import when, col, mean, desc, round, translate


def init_spark():
    spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():
    spark, sc = init_spark()
    # first we load the data into a rdd as proposed by the exercise sheet
    rdd_with_data = sc.textFile("adult.data")

    # after this is done we need to create the columns for our dataframe
    columns_for_dataframe = ["age", "workclass", "flnwgt", "education", "education_number", "marital-status",
                             "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                             "hours-per-week", "native-country", "income"]

    rdd_split_by_delimiter = rdd_with_data.map(lambda l: l.split(","))

    # Convert to Row
    rdd_as_ordered_struct = rdd_split_by_delimiter.map(
        lambda p: Row(age=p[0], workcalss=p[1], flngwt=p[2], education=p[3], educationNumber=p[4], maritalStatus=p[5],
                      occupation=p[6], relationship=p[7], race=p[8], sex=p[9], capitalGain=p[10], capitalLoss=p[11],
                      hoursPerWeek=p[12], nativeCountry=p[13], income=p[14]))

    # now we can go ahead and create the dataframe
    # dataframe_adults = rdd_with_data.toDF(rdd_as_ordered_struct)
    dataframe_adults = spark.createDataFrame(rdd_as_ordered_struct)

    # we need to fix some datatypes
    dataframe_adults = dataframe_adults.withColumn("age", col("age").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("flngwt", col("flngwt").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("educationNumber", col("educationNumber").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("capitalGain", col("capitalGain").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("capitalLoss", col("capitalLoss").cast("double"))
    dataframe_adults = dataframe_adults.withColumn("hoursPerWeek", col("hoursPerWeek").cast("double"))

    # now we will check if the dataframe is in order quickly
    # dataframe_adults.printSchema()
    dataframe_adults.show(20)

    # now we can start to form the wanted queries on the dataframe
    # Q1: rate of males for each type of marital_status
    print("Solving question 1:")
    results_q_1 = dataframe_adults.select(
        dataframe_adults['maritalStatus'],
        # create a 1/0 type col on the fly
        when(col('sex') == ' Male', 1).otherwise(0).alias('is_male')
    )

    # perform the grouping
    results_q_1 = results_q_1.groupBy('maritalStatus').agg(round(mean('is_male'), 2).alias('male_ratio'))
    results_q_1 = results_q_1.orderBy(desc('male_ratio'))
    # show results
    results_q_1.show()

    # Q2: average hours per week of females who make more than 50K ordered by origin country
    print("Solving question 2:")
    results_q_2 = dataframe_adults.select(
        dataframe_adults['nativeCountry'],
        dataframe_adults['hoursPerWeek'],
    ).where(col('sex') == ' Female').where(col('income') == ' >50K')

    # perform the grouping
    results_q_2 = results_q_2.groupBy('nativeCountry').agg({"hoursPerWeek": "avg"})
    results_q_2 = results_q_2.orderBy(desc('avg(hoursPerWeek)'))
    # show results
    results_q_2.show(20)

    # Q3: get highest and lowest education for income groups
    print("Solving question 3:")
    results_q_3 = dataframe_adults.select(
        dataframe_adults['income'],
        dataframe_adults['educationNumber'].alias("enh"),
        dataframe_adults['educationNumber'].alias("enl"),
    )
    # perform the grouping
    results_q_3 = results_q_3.groupBy('income').agg({"enh": "max", "enl": "min"})
    # 1.0 = Preschool + 2.0 = 1st-4th + 16.0 = Doctorate
    results_q_3_final = results_q_3.select(
        results_q_3['income'],
        when(col('min(enl)') == 1.0, 'Preschool').otherwise('1st-4th').alias('lowest_education'),
        when(col('max(enh)') == 16.0, 'Doctorate').otherwise('error').alias('highest_education'),
    )
    # show results
    results_q_3_final.show()


if __name__ == '__main__':
    main()
