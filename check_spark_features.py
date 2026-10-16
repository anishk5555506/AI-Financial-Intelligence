
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("CheckFinancialFeatures")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)

try:
    path = "hdfs://localhost:9000/financial-data/processed"

    df = spark.read.parquet(path)

    print("\nAvailable columns:")
    print(df.columns)

    print("\nRecord count:", df.count())

    print("\nSample feature values:")
    df.filter(df["Ticker"] == "AAPL").select(
        "Date",
        "Ticker",
        "Close",
        "Return_1D",
        "Return_2D",
        "Return_3D",
        "Return_5D",
        "Close_vs_MA20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Position",
        "BB_Width",
        "Target_Return",
        "Target_Price",
    ).orderBy("Date").show(10, truncate=False)

finally:
    spark.stop()

