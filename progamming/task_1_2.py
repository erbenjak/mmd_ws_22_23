from pyspark.sql import SparkSession


def init_spark():
    spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():
    spark, sc = init_spark()

    # -- a) -------------
    # showing some examples on how to use some basic spark functions
    x = sc.parallelize([("a", 1), ("b", 4)])
    y = sc.parallelize([("a", 2), ("a", 3)])

    # the join() function is inspected first:
    result_join = x.join(y)
    # Returns an RDD containing all pairs of elements with matching keys in x and y.
    local_result_join = result_join.collect()
    print("The result of the join operation:")
    print(local_result_join)
    # [('a', (1, 2)), ('a', (1, 3))]
    print("One can observe that the first values of x is combined with each value from printed \n")

    # the sorted() operation is inspected next:
    print("The result of the sorted operation:")
    tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
    print(sc.parallelize(tmp).sortBy(lambda x: x[0]).collect())
    # [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
    print("One can pass a lambda function which is then used for sorting in this case the keys are used\n")

    # the group_by() operation is inspected next:
    numbers_one_to_eight = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8])
    result_group_by = numbers_one_to_eight.groupBy(lambda x: x % 2).collect()
    print("The result of the group_by operation:")
    print(sorted([(x, sorted(y)) for (x, y) in result_group_by]))
    # [(0, [2, 4, 6, 8]), (1, [1, 3, 5, 7])]
    print("One can how we created two collections one passing the lambda check and "
          "one containing all elements which dont.\n")

    # -- b) -------------
    print("We choose the random split as the first transform")
    numbers_one_to_100 = sc.parallelize(range(100), 1)
    res1, res2 = numbers_one_to_100.randomSplit([0.3, 0.7], 17)
    print("length of result 1:" + str(len(res1.collect())))
    print("length of result 2:" + str(len(res2.collect())))
    print("We can observe that we roughly get a random split with the wanted distributions\n")

    print("We choose the union as the second transform")
    first_half = sc.parallelize([('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)])
    second_half = sc.parallelize([('f', 6), ('g', 7), ('h', 8), ('i', 9), ('j', 10)])
    combined_rdd = first_half.union(second_half)
    print(combined_rdd.collect())
    print("We can see that both input rdd got joined together\n")

    print("We choose the intersect as the third transform")
    first_half_int = sc.parallelize([('f', 6), ('g', 7), ('c', 3), ('d', 4), ('e', 5)])
    second_half_int = sc.parallelize([('f', 6), ('g', 7), ('h', 8), ('d', 4), ('e', 5)])
    combined_rdd_int = first_half_int.intersection(second_half_int)
    print(combined_rdd_int.collect())
    print("We can see that both input rdd got intersected together\n")

    # --- Actions
    print("We will show the following actions: first() / take() / count()")
    print("Therefore the last result will be used")
    print("First:")
    print(combined_rdd_int.first())
    print("Take(3):")
    print(combined_rdd_int.take(3))
    print("Count:")
    print(combined_rdd_int.count())


if __name__ == '__main__':
    main()
