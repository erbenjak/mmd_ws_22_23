from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import when, col, mean, desc, round, translate, isnull, count, row_number, monotonically_increasing_id
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType
from pyspark.sql.window import Window

from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry, IndexedRowMatrix, IndexedRow
from pyspark.mllib.linalg import SparseVector

from pyspark.ml.feature import VectorAssembler



class SparseVectorHelper:
    def __init__(self, size:int, indices:list(int), values:list(int)):
        self.size = size
        self.indices = indices
        self.values = values
      
    def appendEntry(self, index: int, value:int) -> None:
        self.indices.append(index)
        self.values.append(value)

    def toSparseVector(self) -> SparseVector:
        return SparseVector(self.size, self.indices, self.values)


def init_spark() -> None:
    spark = SparkSession.builder.appName("Exercise_2_5").getOrCreate()
    sc = spark.sparkContext
    return spark, sc

def fill_sparse_vector(user:Row, vector:SparseVectorHelper) -> None:
    SparseVectorHelper.appendEntry(user)
    print(user["index_user"])


def main():
    spark, sc = init_spark()

    # schema for user artist
    schema_df_user_artist = StructType([
            StructField ("userid", StringType(), True), 
            StructField ("artistid",  StringType(), True),
            StructField ("playcount", IntegerType(), True)
        ])

    # read data
    df_user_artist = spark.read.format("csv")\
                .option("header", "false")\
                .option("sep"," ")\
                .schema(schema_df_user_artist)\
                .load("user_artist_data_small.txt")
    df_user_artist.show(5)

    # schema for artist_alias
    schema_df_artist_alias = StructType([
            StructField ("wrong_artist", StringType(), True), 
            StructField ("right_artist",  StringType(), True),
        ])

    # read alias data
    df_artist_alias = spark.read.format("csv")\
                .option("header", "false")\
                .option("sep","\t")\
                .schema(schema_df_artist_alias)\
                .load("artist_alias_small.txt")
    df_artist_alias.show(5)
    
    # replace the wrong artist ids by correct ones by joining them to the original dataset and the replacing null values
    df_joined = df_user_artist\
                .alias("l")\
                .join(df_artist_alias.alias("r"), col("l.artistid")==col("r.wrong_artist"), how='left')\
                .select(
                    col("l.userid").alias("userid"),
                    when(isnull(col('r.right_artist')), 
                        col("l.artistid"))
                        .otherwise(col('r.right_artist'))
                        .alias('artistid'),
                    col("l.playcount").alias("playcount"))\
                .distinct()
    df_joined.show(5)

    # get dataframe with one column of distinct users
    df_distinct_users = df_joined.select('userid').distinct().sort(col('userid'))
    df_distinct_users = df_distinct_users.withColumn("index", row_number().over(Window.orderBy(monotonically_increasing_id()))-1)
    df_distinct_users.show(5)
    # build index for users
    user_count = df_distinct_users.count()
    print(user_count)

    # get dataframe with one column of distinct artists
    df_distinct_artists = df_joined.select('artistid').distinct().sort(col('artistid'))
    # build index for artists
    df_distinct_artists = df_distinct_artists.withColumn("index", row_number().over(Window.orderBy(monotonically_increasing_id()))-1)
    df_distinct_artists.show(5)
    artist_count = df_distinct_artists.count()
    print(artist_count)

    # join index 
    df_joined = df_joined\
                .join(df_distinct_artists, "artistid", how='inner')\
                .select(df_joined["*"], col("index").alias("index_artist"))

    df_joined = df_joined\
            .join(df_distinct_users, "userid", how='inner')\
            .select(df_joined["*"], col("index").alias("index_user"))\
            .sort("index_user")
    df_joined.show(5)

    # idea: use sparse vector helper class to build a list of a sparse vector 
    # representation and then use that as input for a real sparse vector for each user
    # then we could use this datastructure and the norm function to compute correlation 
    # coefficient pairwise by iterating over all possible pairs of sparse vectors 


if __name__ == '__main__':
    main()