import os
from pathlib import Path

import oracledb
import pandas as pd
from dotenv import load_dotenv


# Load environment variables from the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def fetch_stock_history(ticker):

    password = os.getenv("ORACLE_PASSWORD")

    if not password:
        raise RuntimeError(
            "ORACLE_PASSWORD is missing. Check your .env file."
        )

    connection = oracledb.connect(
        user=os.getenv("ORACLE_USER", "FINANCE_USER"),
        password=password,
        dsn=os.getenv("ORACLE_DSN", "localhost:1521/XEPDB1")
    )

    try:
        cursor = connection.cursor()

        try:
            cursor.execute("""
                SELECT
                    TRADE_DATE,
                    OPEN_PRICE,
                    HIGH_PRICE,
                    LOW_PRICE,
                    CLOSE_PRICE,
                    VOLUME
                FROM STOCK_HISTORY
                WHERE TICKER = :ticker
                ORDER BY TRADE_DATE
            """, {"ticker": ticker})

            rows = cursor.fetchall()

        finally:
            cursor.close()

    finally:
        connection.close()

    columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    data = pd.DataFrame(rows, columns=columns)

    if data.empty:
        return data

    data["Date"] = pd.to_datetime(data["Date"])

    numeric_columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    data = data.dropna(subset=numeric_columns)
    data = data[data["Close"] > 0]

    data = data.sort_values("Date").reset_index(drop=True)

    return data