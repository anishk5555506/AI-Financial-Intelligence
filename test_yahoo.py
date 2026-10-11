import yfinance as yf

data = yf.download(
    "AAPL",
    period="2y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if data.empty:
    print("No data received.")
else:
    print(data.tail())
    print("Total records:", len(data))