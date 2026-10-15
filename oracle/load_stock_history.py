import getpass
import pandas as pd
import yfinance as yf
import oracledb


# Stock symbols used by our existing Flask application
TICKERS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla"
}


# Insert new records or update existing records.
# This prevents duplicate entries when the script runs again.
SQL = """
MERGE INTO STOCK_HISTORY target
USING (
    SELECT
        :ticker AS TICKER,
        :company_name AS COMPANY_NAME,
        :trade_date AS TRADE_DATE,
        :open_price AS OPEN_PRICE,
        :high_price AS HIGH_PRICE,
        :low_price AS LOW_PRICE,
        :close_price AS CLOSE_PRICE,
        :volume AS VOLUME
    FROM dual
) source
ON (
    target.TICKER = source.TICKER
    AND target.TRADE_DATE = source.TRADE_DATE
)
WHEN MATCHED THEN
    UPDATE SET
        target.COMPANY_NAME = source.COMPANY_NAME,
        target.OPEN_PRICE = source.OPEN_PRICE,
        target.HIGH_PRICE = source.HIGH_PRICE,
        target.LOW_PRICE = source.LOW_PRICE,
        target.CLOSE_PRICE = source.CLOSE_PRICE,
        target.VOLUME = source.VOLUME
WHEN NOT MATCHED THEN
    INSERT (
        TICKER, COMPANY_NAME, TRADE_DATE,
        OPEN_PRICE, HIGH_PRICE, LOW_PRICE,
        CLOSE_PRICE, VOLUME
    )
    VALUES (
        source.TICKER, source.COMPANY_NAME,
        source.TRADE_DATE, source.OPEN_PRICE,
        source.HIGH_PRICE, source.LOW_PRICE,
        source.CLOSE_PRICE, source.VOLUME
    )
"""


def download_stock_data(ticker, company_name):

    print(f"\nDownloading data for {company_name}...")

    data = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No data received for {ticker}")

    # Handle the MultiIndex format used by some yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    required = [
        "Date", "Open", "High",
        "Low", "Close", "Volume"
    ]

    data = data[required].copy()

    numeric_columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    data = data.dropna(subset=numeric_columns)
    data = data[data["Close"] > 0]
    data = data.sort_values("Date")

    records = []

    for _, row in data.iterrows():

        trade_date = pd.Timestamp(row["Date"]).to_pydatetime()

        # Oracle DATE values here represent a trading date,
        # so remove timezone information and normalize the time.
        if trade_date.tzinfo is not None:
            trade_date = trade_date.replace(tzinfo=None)

        trade_date = trade_date.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        records.append({
            "ticker": ticker,
            "company_name": company_name,
            "trade_date": trade_date,
            "open_price": float(row["Open"]),
            "high_price": float(row["High"]),
            "low_price": float(row["Low"]),
            "close_price": float(row["Close"]),
            "volume": int(row["Volume"])
        })

    print(f"Prepared {len(records)} records for {ticker}")

    return records


def main():

    password = getpass.getpass(
        "Enter FINANCE_USER password: "
    )

    connection = None
    cursor = None

    try:

        # Download data before opening the database transaction.
        all_records = []

        for ticker, company_name in TICKERS.items():
            records = download_stock_data(
                ticker, company_name
            )

            all_records.extend(records)

        # Connect to our Oracle database.
        connection = oracledb.connect(
            user="FINANCE_USER",
            password=password,
            dsn="localhost:1521/XEPDB1"
        )

        cursor = connection.cursor()

        # Insert or update the records in batches.
        cursor.executemany(SQL, all_records)

        connection.commit()

        print("\nStock data loaded successfully!")
        print(f"Total records processed: {len(all_records)}")

    except Exception as error:

        if connection:
            connection.rollback()

        print("\nError loading stock data:", error)

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


if __name__ == "__main__":
    main()