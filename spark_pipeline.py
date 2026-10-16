
import argparse
import logging
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
}

HDFS_RAW = "hdfs://localhost:9000/financial-data/raw"
HDFS_PROCESSED = "hdfs://localhost:9000/financial-data/processed"

LOCAL_RAW_DIR = "data/spark_raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def create_spark_session():
    """Create a Spark session running locally."""
    return (
        SparkSession.builder
        .appName("FinancialAIDataPipeline")
        .master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


def download_stock_data():
    """Download historical stock prices and save raw CSV files."""
    os.makedirs(LOCAL_RAW_DIR, exist_ok=True)

    downloaded_files = []

    for company, ticker in TICKERS.items():
        logger.info("Downloading %s (%s)", company, ticker)

        data = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            logger.warning("No data returned for %s", ticker)
            continue

        # Support different yfinance column formats.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        required = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        missing = [column for column in required if column not in data.columns]

        if missing:
            logger.warning(
                "Skipping %s because columns are missing: %s",
                ticker,
                missing,
            )
            continue

        data = data[required].copy()
        data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

        for column in ["Open", "High", "Low", "Close", "Volume"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        data["Ticker"] = ticker
        data["Company"] = company
        data["Collected_At"] = datetime.now(timezone.utc).isoformat()

        filename = os.path.join(LOCAL_RAW_DIR, f"{ticker}.csv")
        data.to_csv(filename, index=False)

        downloaded_files.append((ticker, filename))

        logger.info(
            "Saved %d records for %s to %s",
            len(data),
            ticker,
            filename,
        )

    if not downloaded_files:
        raise RuntimeError(
            "No stock data was downloaded. Check your internet connection "
            "and Yahoo Finance availability."
        )

    return downloaded_files


def upload_raw_data_to_hdfs(downloaded_files):
    """Upload each local CSV to HDFS."""
    import subprocess

    subprocess.run(
        ["hdfs", "dfs", "-mkdir", "-p", HDFS_RAW],
        check=True,
    )

    for ticker, filename in downloaded_files:
        destination = f"{HDFS_RAW}/{ticker}.csv"

        subprocess.run(
            ["hdfs", "dfs", "-put", "-f", filename, destination],
            check=True,
        )

        logger.info("Uploaded %s to %s", ticker, destination)


def process_with_spark(spark):
    """Read stock data from HDFS, clean it, and calculate features."""
    logger.info("Reading raw stock data from HDFS")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{HDFS_RAW}/*.csv")
    )

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Ticker",
        "Company",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in HDFS data: {missing}")

    # Parse dates and enforce numeric types.
    df = (
        df.withColumn("Date", F.to_date("Date"))
        .withColumn("Open", F.col("Open").cast("double"))
        .withColumn("High", F.col("High").cast("double"))
        .withColumn("Low", F.col("Low").cast("double"))
        .withColumn("Close", F.col("Close").cast("double"))
        .withColumn("Volume", F.col("Volume").cast("double"))
    )

    # Remove invalid records and duplicate ticker/date combinations.
    df = (
        df.dropna(
            subset=[
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Ticker",
            ]
        )
        .filter(
            (F.col("Open") > 0)
            & (F.col("High") > 0)
            & (F.col("Low") > 0)
            & (F.col("Close") > 0)
            & (F.col("Volume") >= 0)
        )
        .dropDuplicates(["Ticker", "Date"])
    )

    window = (
        Window.partitionBy("Ticker")
        .orderBy("Date")
    )

    window_5 = window.rowsBetween(-4, 0)
    window_10 = window.rowsBetween(-9, 0)
    window_20 = window.rowsBetween(-19, 0)
    window_50 = window.rowsBetween(-49, 0)

    # Historical returns.
    df = (
        df.withColumn(
            "Previous_Close",
            F.lag("Close", 1).over(window),
        )
        .withColumn(
            "Return_1D",
            F.col("Close") / F.col("Previous_Close") - 1,
        )
        .withColumn(
            "Return_2D",
            F.col("Close") / F.lag("Close", 2).over(window) - 1,
        )
	.withColumn(
		"Return_3D",
    		F.col("Close") / F.lag("Close", 3).over(window) - 1,
	)
        .withColumn(
            "Return_5D",
            F.col("Close") / F.lag("Close", 5).over(window) - 1,
        )
        .withColumn(
            "Return_10D",
            F.col("Close") / F.lag("Close", 10).over(window) - 1,
        )
        .withColumn(
            "Return_20D",
            F.col("Close") / F.lag("Close", 20).over(window) - 1,
        )
    )

    # Lagged returns.
    for lag in [1, 2, 3, 5, 10]:
        df = df.withColumn(
            f"Return_Lag_{lag}",
            F.lag("Return_1D", lag).over(window),
        )

    # Moving averages and normalized distances.
    for period, rolling_window in [
        (5, window_5),
        (10, window_10),
        (20, window_20),
        (50, window_50),
    ]:
        df = df.withColumn(
            f"MA_{period}",
            F.avg("Close").over(rolling_window),
        ).withColumn(
            f"Close_vs_MA{period}",
            F.col("Close") / F.col(f"MA_{period}") - 1,
        )

    # Rolling return volatility.
    for period, rolling_window in [
        (5, window_5),
        (10, window_10),
        (20, window_20),
    ]:
        df = df.withColumn(
            f"Volatility_{period}D",
            F.stddev_samp("Return_1D").over(rolling_window),
        )

    # RSI approximation using rolling average gains and losses.
    delta = F.col("Close") - F.col("Previous_Close")

    df = (
        df.withColumn("Price_Change", delta)
        .withColumn(
            "Gain",
            F.when(F.col("Price_Change") > 0, F.col("Price_Change"))
            .otherwise(0.0),
        )
        .withColumn(
            "Loss",
            F.when(F.col("Price_Change") < 0, -F.col("Price_Change"))
            .otherwise(0.0),
        )
        .withColumn(
            "Avg_Gain_14",
            F.avg("Gain").over(window.rowsBetween(-13, 0)),
        )
        .withColumn(
            "Avg_Loss_14",
            F.avg("Loss").over(window.rowsBetween(-13, 0)),
        )
    )

    relative_strength = F.when(
        F.col("Avg_Loss_14") > 0,
        F.col("Avg_Gain_14") / F.col("Avg_Loss_14"),
    ).otherwise(None)



    # RSI_14 is calculated inside add_advanced_indicators below.
    # Initialize the column here so Spark's output schema includes it.
    df = df.withColumn(
        "RSI_14",
        F.lit(None).cast("double"),
    )
    # MACD based on exponential moving averages.
    # Use a grouped pandas calculation for the exponential indicators.
    # This avoids pretending a simple moving average is an EMA.
    from pyspark.sql.types import (
        DoubleType,
        StructField,
        StructType,
    )

    schema = StructType(
        df.schema.fields
        + [
            StructField("EMA_12", DoubleType(), True),
            StructField("EMA_26", DoubleType(), True),
            StructField("MACD", DoubleType(), True),
            StructField("MACD_Signal", DoubleType(), True),
            StructField("MACD_Histogram", DoubleType(), True),
            StructField("BB_Position", DoubleType(), True),
            StructField("BB_Width", DoubleType(), True),
        ]
    )

    base_columns = df.columns

    def add_advanced_indicators(pdf):
        pdf = pdf.sort_values("Date").copy()

        close = pdf["Close"]
	        # RSI calculation matching model.py exactly.
        delta = close.diff()

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()

        avg_loss = losses.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()

        relative_strength = avg_gain / avg_loss.replace(0, np.nan)

        rsi = 100 - (100 / (1 + relative_strength))

        rsi = rsi.mask(
            (avg_loss == 0) & (avg_gain > 0),
            100,
        )

        rsi = rsi.mask(
            (avg_loss == 0) & (avg_gain == 0),
            50,
        )

        pdf["RSI_14"] = rsi / 100.0
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        macd_raw = ema12 - ema26
        signal_raw = macd_raw.ewm(span=9, adjust=False).mean()

        pdf["EMA_12"] = ema12
        pdf["EMA_26"] = ema26
        pdf["MACD"] = macd_raw / close
        pdf["MACD_Signal"] = signal_raw / close
        pdf["MACD_Histogram"] = (macd_raw - signal_raw) / close

        middle = close.rolling(20).mean()
        standard_deviation = close.rolling(20).std()

        upper = middle + 2 * standard_deviation
        lower = middle - 2 * standard_deviation

        width = upper - lower

        pdf["BB_Position"] = (close - lower) / width
        pdf["BB_Width"] = width / middle

        return pdf

    # applyInPandas requires PyArrow.
    import pyarrow  # noqa: F401

    df = (
        df.groupBy("Ticker")
        .applyInPandas(add_advanced_indicators, schema=schema)
    )

    # Additional price and volume features.
    df = (
        df.withColumn(
            "Intraday_Range",
            (F.col("High") - F.col("Low")) / F.col("Close"),
        )
        .withColumn(
            "Open_Gap",
            F.col("Open") / F.col("Previous_Close") - 1,
        )
        .withColumn(
            "Close_vs_Open",
            F.col("Close") / F.col("Open") - 1,
        )
        .withColumn(
            "Volume_Change",
            F.col("Volume") / F.lag("Volume", 1).over(window) - 1,
        )
        .withColumn(
            "Volume_MA20",
            F.avg("Volume").over(window_20),
        )
        .withColumn(
            "Volume_vs_MA20",
            F.col("Volume") / F.col("Volume_MA20") - 1,
        )
        .withColumn(
            "Volume_MA5",
            F.avg("Volume").over(window_5),
        )
        .withColumn(
            "Volume_vs_MA5",
            F.col("Volume") / F.col("Volume_MA5") - 1,
        )
        .withColumn(
            "High_20D",
            F.max("High").over(window_20),
        )
        .withColumn(
            "Low_20D",
            F.min("Low").over(window_20),
        )
        .withColumn(
            "High_vs_20D",
            F.col("Close") / F.col("High_20D") - 1,
        )
        .withColumn(
            "Low_vs_20D",
            F.col("Close") / F.col("Low_20D") - 1,
        )
        .withColumn(
            "Target_Return",
            F.log(
                F.lead("Close", 1).over(window) / F.col("Close")
            ),
        )
        .withColumn(
            "Target_Price",
            F.lead("Close", 1).over(window),
        )
        .withColumn(
            "Target_Date",
            F.lead("Date", 1).over(window),
        )
    )

    logger.info("Saving processed data to HDFS")

    (
        df.write
        .mode("overwrite")
        .partitionBy("Ticker")
        .parquet(HDFS_PROCESSED)
    )

    logger.info("Processed record count: %d", df.count())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Financial AI Hadoop and Spark data pipeline"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse local raw CSV files and upload them to HDFS",
    )
    args = parser.parse_args()

    spark = create_spark_session()

    try:
        if args.skip_download:
            downloaded_files = [
                (ticker, os.path.join(LOCAL_RAW_DIR, f"{ticker}.csv"))
                for ticker in TICKERS.values()
                if os.path.exists(
                    os.path.join(LOCAL_RAW_DIR, f"{ticker}.csv")
                )
            ]

            if not downloaded_files:
                raise RuntimeError(
                    "No local CSV files found. Run without --skip-download."
                )
        else:
            downloaded_files = download_stock_data()

        upload_raw_data_to_hdfs(downloaded_files)

        processed = process_with_spark(spark)

        processed.select(
            "Date",
            "Ticker",
            "Close",
            "Return_1D",
            "Close_vs_MA20",
            "RSI_14",
            "MACD",
            "Target_Price",
        ).orderBy("Ticker", "Date").show(15, truncate=False)

        logger.info("Pipeline completed successfully")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

