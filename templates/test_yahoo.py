import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import binomtest, norm
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
 
warnings.filterwarnings("ignore")
 
# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
TARGET = "AAPL"
CONTEXT = ["^GSPC", "^VIX", "XLK"]   # for Indian stocks use e.g. ["^NSEI", "^INDIAVIX"]
START = "2010-01-01"
MIN_TRAIN = 750        # first training window (~3 years)
TEST_SIZE = 60         # each walk-forward fold tests on the next ~3 months
SEED = 42
 
 
# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def load_data():
    tickers = [TARGET] + CONTEXT
    raw = yf.download(tickers, start=START, interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("No data received from Yahoo Finance.")
 
    close = raw["Close"].copy()
    volume = raw["Volume"][TARGET].copy()
 
    close = close.dropna(subset=[TARGET])
    close = close.ffill().dropna()          # forward-fill index gaps only
    volume = volume.reindex(close.index).ffill()
    print(f"Downloaded {len(close)} rows: {close.index[0].date()} -> {close.index[-1].date()}")
    return close, volume
 
 
# ----------------------------------------------------------------------------
# 2. Features (every feature at row t uses only data up to close of day t)
# ----------------------------------------------------------------------------
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)
 
 
def build_dataset(close, volume):
    px = close[TARGET]
    r = np.log(px).diff() * 100            # daily log return in percent
    X = pd.DataFrame(index=close.index)
 
    # Lagged returns
    for k in (0, 1, 2, 3, 4, 9):
        X[f"ret_lag{k}"] = r.shift(k)
 
    # Momentum / trend
    X["ret_5d"] = r.rolling(5).sum()
    X["ret_20d"] = r.rolling(20).sum()
    for w in (10, 20, 50):
        X[f"px_sma{w}"] = px / px.rolling(w).mean() - 1
    X["rsi14"] = rsi(px)
    ema12, ema26 = px.ewm(span=12, adjust=False).mean(), px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    X["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / px * 100
 
    # Volatility
    X["vol_5"] = r.rolling(5).std()
    X["vol_20"] = r.rolling(20).std()
    X["vol_ratio"] = X["vol_5"] / X["vol_20"]
 
    # Volume
    X["volume_ratio"] = volume / volume.rolling(20).mean()
 
    # Market context
    if "^GSPC" in close:
        spx = np.log(close["^GSPC"]).diff() * 100
        X["spx_ret0"], X["spx_ret1"] = spx, spx.shift(1)
    if "XLK" in close:
        xlk = np.log(close["XLK"]).diff() * 100
        X["xlk_ret0"], X["xlk_ret1"] = xlk, xlk.shift(1)
    if "^VIX" in close:
        vix = close["^VIX"]
        X["vix_level"] = vix
        X["vix_chg"] = vix.pct_change() * 100
        X["vix_rel20"] = vix / vix.rolling(20).mean()
 
    # Calendar
    X["dow"] = X.index.dayofweek
 
    latest = X.iloc[[-1]].copy()           # today's features -> tomorrow's forecast
    latest_close = px.iloc[-1]
 
    df = X.copy()
    df["y"] = r.shift(-1)                  # TARGET: next-day return (%)
    df["close"] = px
    df["next_close"] = px.shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, list(X.columns), latest, latest_close
 
 
# ----------------------------------------------------------------------------
# 3. Models (scaling is inside the pipeline -> fit on training data only)
# ----------------------------------------------------------------------------
def make_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=20,
            max_features=0.5, n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=200,
            min_samples_leaf=30, l2_regularization=1.0, random_state=SEED),
        "KNN Regressor": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=30)),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "Support Vector Regression": make_pipeline(StandardScaler(), SVR(C=0.5, epsilon=0.1)),
    }
 
 
# ----------------------------------------------------------------------------
# 4. Walk-forward validation
# ----------------------------------------------------------------------------
def walk_forward(df, features):
    n = len(df)
    folds = list(range(MIN_TRAIN, n - 1, TEST_SIZE))
    store = {name: [] for name in make_models()}
    store["Ensemble (avg of 5)"] = []
    fold_id, truth = [], []
 
    for i, s in enumerate(folds):
        train = df.iloc[:s]
        test = df.iloc[s:s + TEST_SIZE]
        preds = {}
        for name, model in make_models().items():
            model.fit(train[features], train["y"])
            preds[name] = model.predict(test[features])
        preds["Ensemble (avg of 5)"] = np.mean(list(preds.values()), axis=0)
 
        for name, p in preds.items():
            store[name].append(p)
        fold_id.append(np.full(len(test), i))
        truth.append(test[["y", "close", "next_close"]])
 
    truth = pd.concat(truth)
    fold_id = np.concatenate(fold_id)
    store = {k: np.concatenate(v) for k, v in store.items()}
    return store, truth, fold_id
 
 
# ----------------------------------------------------------------------------
# 5. Evaluation
# ----------------------------------------------------------------------------
def mae(e): return float(np.mean(np.abs(e)))
def rmse(e): return float(np.sqrt(np.mean(np.square(e))))
 
 
def diebold_mariano(err_base, err_model):
    """One-sided: is the model's squared error lower than the baseline's?"""
    d = np.square(err_base) - np.square(err_model)
    stat = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
    return float(1 - norm.cdf(stat))       # small p => model genuinely better
 
 
def evaluate(store, truth, fold_id):
    actual_price = truth["next_close"].values
    base_price = truth["close"].values     # baseline: tomorrow = today
    base_err = actual_price - base_price
    actual_dir = np.sign(truth["y"].values)
 
    rows = []
    for name, pred_ret in store.items():
        pred_price = truth["close"].values * np.exp(pred_ret / 100)
        err = actual_price - pred_price
 
        hits = int((np.sign(pred_ret) == actual_dir).sum())
        n = len(pred_ret)
        p_dir = binomtest(hits, n, 0.5).pvalue
 
        beats = [mae(err[fold_id == f]) < mae(base_err[fold_id == f])
                 for f in np.unique(fold_id)]
 
        rows.append({
            "Model": name,
            "MAE": mae(err),
            "RMSE": rmse(err),
            "Baseline MAE": mae(base_err),
            "Baseline RMSE": rmse(base_err),
            "Beats baseline?": mae(err) < mae(base_err) and rmse(err) < rmse(base_err),
            "Direction acc %": 100 * hits / n,
            "Direction p-value": p_dir,
            "DM p-value (vs base)": diebold_mariano(base_err, err),
            "Folds beating base %": 100 * np.mean(beats),
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
 
 
# ----------------------------------------------------------------------------
# 6. Tomorrow's forecast (fit on all history)
# ----------------------------------------------------------------------------
def forecast_tomorrow(df, features, latest, latest_close):
    preds = {}
    for name, model in make_models().items():
        model.fit(df[features], df["y"])
        preds[name] = float(model.predict(latest[features])[0])
    preds["Ensemble (avg of 5)"] = float(np.mean(list(preds.values())))
    out = pd.DataFrame({
        "Model": list(preds.keys()),
        "Predicted return %": list(preds.values()),
        "Predicted price": [latest_close * np.exp(v / 100) for v in preds.values()],
    })
    return out
 
 
# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    close, volume = load_data()
    df, features, latest, latest_close = build_dataset(close, volume)
    print(f"Modelling rows: {len(df)} | features: {len(features)}")
 
    store, truth, fold_id = walk_forward(df, features)
    print(f"Walk-forward: {len(np.unique(fold_id))} folds, {len(truth)} test days\n")
 
    results = evaluate(store, truth, fold_id)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(results.round(4).to_string(index=False))
    results.to_csv("model_results.csv", index=False)
 
    print("\nHow to read this:")
    print(" - Beats baseline? needs both MAE and RMSE lower than 'tomorrow = today'.")
    print(" - Direction p-value < 0.05 means direction accuracy is unlikely to be luck.")
    print(" - DM p-value < 0.05 means the model's errors are significantly lower than baseline.")
    print(" - Folds beating base % shows consistency across time periods.")
 
    print("\nForecast for the next trading day:")
    print(forecast_tomorrow(df, features, latest, latest_close).round(3).to_string(index=False))
 