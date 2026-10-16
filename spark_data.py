import logging
from functools import lru_cache

import pandas as pd
from pyspark.sql import SparkSession


HDFS_PROCESSED = "hdfs://localhost:9000/financial-data/processed"

TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
}

logger = logging.getLogger(__name__)

_spark = None


def get_spark_session():
    """Create one reusable Spark session."""

    global _spark

    if _spark is None:
        _spark = (
            SparkSession.builder
            .appName("FinancialAIFlaskIntegration")
            .master("local[2]")
            .config("spark.driver.memory", "1g")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )

    return _spark


@lru_cache(maxsize=3)
def _load_company_data(ticker):
    """Read a company's historical OHLCV data from HDFS."""

    spark = get_spark_session()

    df = spark.read.parquet(HDFS_PROCESSED)

    company_df = (
        df.filter(df.Ticker == ticker)
        .select(
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        )
        .orderBy("Date")
    )

    pandas_df = company_df.toPandas()

    if pandas_df.empty:
        raise ValueError(
            f"No historical data found for {ticker} in HDFS."
        )

    pandas_df["Date"] = pd.to_datetime(pandas_df["Date"])

    logger.info(
        "Loaded %d records for %s from HDFS",
        len(pandas_df),
        ticker,
    )

    return pandas_df


def get_stock_data(company):
    """Return historical stock data for a supported company."""

    ticker = TICKERS.get(company)

    if ticker is None:
        raise ValueError(f"Unsupported company: {company}")

    return _load_company_data(ticker).copy()
