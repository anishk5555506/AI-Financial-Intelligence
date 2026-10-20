import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


MODELS = {
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "linear_regression": "Ridge Regression",
    "knn": "KNN Regressor",
    "svr": "Support Vector Regression",
}

HORIZONS = (1, 5, 10)            # price and range forecasts (trading days)
VOL_HORIZONS = (5, 10, 20)       # volatility forecasts (trading days)

ANNUALIZATION = np.sqrt(252) * 100   # daily log-return -> annualized %

BASE_FEATURES = [
    "Return_1D", "Return_2D", "Return_3D", "Return_5D",
    "Return_10D", "Return_20D",
    "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
    "Return_Lag_5", "Return_Lag_10",
    "Close_vs_MA5", "Close_vs_MA10", "Close_vs_MA20",
    "Close_vs_MA50",
    "Volatility_5D", "Volatility_10D", "Volatility_20D",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Histogram",
    "BB_Position", "BB_Width",
    "Intraday_Range", "Open_Gap", "Close_vs_Open",
    "Volume_Change", "Volume_vs_MA20", "Volume_vs_MA5",
    "High_vs_20D", "Low_vs_20D",
]

WF_TEST_SIZE = 60      # each walk-forward fold tests on ~3 months
WF_MAX_SPLITS = 8
WF_MIN_TRAIN = 120

# Prediction ranges
RANGE_LEVEL = 0.90                    # 90% range
LO_Q = (1 - RANGE_LEVEL) / 2          # 5th percentile
HI_Q = 1 - LO_Q                       # 95th percentile
METHOD_NAMES = {
    "ml": "Quantile Gradient Boosting",
    "vol_scaled": "Volatility-scaled history",
}


def _horizon_label(horizon):
    return (
        "next trading day" if horizon == 1
        else f"{horizon} trading days"
    )


# --------------------------------------------------
# MODEL CREATION
# --------------------------------------------------

def create_model(model_id):

    if model_id == "random_forest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=20,
            max_features=0.5,
            random_state=42,
            n_jobs=-1,
        )

    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=3,
            max_leaf_nodes=8,
            min_samples_leaf=30,
            l2_regularization=2.0,
            learning_rate=0.03,
            random_state=42,
        )

    if model_id == "linear_regression":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=100.0),
        )

    if model_id == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=30,
                weights="uniform",
            ),
        )

    if model_id == "svr":
        return TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=0.1, epsilon=0.1),
            ),
            transformer=StandardScaler(),
        )

    raise ValueError(f"Unknown model: {model_id}")


def create_quantile_model(quantile):
    """Gradient boosting trained to predict one percentile of the outcome."""
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=100,
        max_depth=2,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=42,
    )


# --------------------------------------------------
# RSI INDICATOR
# --------------------------------------------------

def _rsi(close, period=14):

    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))

    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100)
    result = result.mask((avg_loss == 0) & (avg_gain == 0), 50)

    return result


# --------------------------------------------------
# FEATURE ENGINEERING (shared by price, range and volatility models)
# --------------------------------------------------

def _build_features(data):
    """
    Returns (df, base_columns, volatility_columns).
    Every feature on row t only uses data up to the close of day t.
    """

    if data is None or data.empty:
        raise ValueError("No historical data was supplied.")

    df = data.copy()

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric = ["Open", "High", "Low", "Close", "Volume"]
    for optional in ("SPX_Close", "VIX_Close"):
        if optional in df.columns:
            numeric.append(optional)

    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    df = df[(df["Close"] > 0) & (df["Open"] > 0)].reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("At least 100 historical rows are required.")

    close = df["Close"]
    volume = df["Volume"].clip(lower=0)
    returns = close.pct_change()

    # Historical returns
    df["Return_1D"] = close.pct_change(1)
    for period in [2, 3, 5, 10, 20]:
        df[f"Return_{period}D"] = close.pct_change(period)

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f"Return_Lag_{lag}"] = returns.shift(lag)

    # Moving-average relationships
    for window in [5, 10, 20, 50]:
        df[f"Close_vs_MA{window}"] = close / close.rolling(window).mean() - 1

    # Rolling volatility
    for window in [5, 10, 20]:
        df[f"Volatility_{window}D"] = returns.rolling(window).std()

    # RSI
    df["RSI_14"] = _rsi(close) / 100.0

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_difference = ema12 - ema26
    macd_signal = macd_difference.ewm(span=9, adjust=False).mean()

    df["MACD"] = macd_difference / close
    df["MACD_Signal"] = macd_signal / close
    df["MACD_Histogram"] = (macd_difference - macd_signal) / close

    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20

    df["BB_Position"] = (
        (close - lower_band) / (upper_band - lower_band).replace(0, np.nan)
    )
    df["BB_Width"] = (upper_band - lower_band) / ma20

    # Intraday price information
    previous_close = close.shift(1)
    df["Intraday_Range"] = (df["High"] - df["Low"]) / close
    df["Open_Gap"] = df["Open"] / previous_close - 1
    df["Close_vs_Open"] = close / df["Open"] - 1

    # Volume features
    df["Volume_Change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    for window in [5, 20]:
        volume_ma = volume.rolling(window).mean()
        df[f"Volume_vs_MA{window}"] = volume / volume_ma.replace(0, np.nan) - 1

    # Price position relative to recent extremes
    df["High_vs_20D"] = close / df["High"].rolling(20).max() - 1
    df["Low_vs_20D"] = close / df["Low"].rolling(20).min() - 1

    # Market context (optional)
    base_columns = list(BASE_FEATURES)
    has_spx = (
        "SPX_Close" in df.columns and df["SPX_Close"].notna().mean() > 0.9
    )
    has_vix = (
        "VIX_Close" in df.columns and df["VIX_Close"].notna().mean() > 0.9
    )

    if has_spx:
        spx = df["SPX_Close"]
        df["SPX_Return_1D"] = spx.pct_change(1)
        df["SPX_Return_5D"] = spx.pct_change(5)
        base_columns += ["SPX_Return_1D", "SPX_Return_5D"]

    if has_vix:
        vix = df["VIX_Close"]
        df["VIX_Level"] = vix / 100.0
        df["VIX_Change"] = vix.pct_change(1)
        df["VIX_vs_MA20"] = vix / vix.rolling(20).mean() - 1
        base_columns += ["VIX_Level", "VIX_Change", "VIX_vs_MA20"]

    # Extra features used only by the volatility models
    log_returns = np.log(close).diff()
    squared = log_returns ** 2
    volatility_columns = []

    for window in (5, 10, 20, 60):
        realized = np.sqrt(squared.rolling(window).mean()) * ANNUALIZATION
        df[f"LogRV_{window}"] = np.log(realized.clip(lower=1e-4))
        volatility_columns.append(f"LogRV_{window}")

    df["Abs_Ret_1D"] = log_returns.abs() * 100
    df["Abs_Ret_Lag_1"] = df["Abs_Ret_1D"].shift(1)
    df["Range_MA5"] = df["Intraday_Range"].rolling(5).mean()
    df["Range_MA20"] = df["Intraday_Range"].rolling(20).mean()
    volatility_columns += [
        "Abs_Ret_1D", "Abs_Ret_Lag_1", "Range_MA5", "Range_MA20",
    ]

    if has_spx:
        spx_log = np.log(df["SPX_Close"]).diff()
        spx_rv = np.sqrt((spx_log ** 2).rolling(20).mean()) * ANNUALIZATION
        df["SPX_LogRV_20"] = np.log(spx_rv.clip(lower=1e-4))
        volatility_columns.append("SPX_LogRV_20")

    all_columns = base_columns + volatility_columns
    df[all_columns] = df[all_columns].replace([np.inf, -np.inf], np.nan)

    return df, base_columns, volatility_columns


# --------------------------------------------------
# DATA PREPARATION: PRICE / RANGE MODELS
# --------------------------------------------------

def prepare_data(data, horizon=1):

    if horizon not in HORIZONS:
        raise ValueError(f"Horizon must be one of {HORIZONS}.")

    df, feature_columns, _ = _build_features(data)
    close = df["Close"]

    df["Target_Return"] = np.log(close.shift(-horizon) / close)
    df["Target_Price"] = close.shift(-horizon)
    df["Target_Date"] = df["Date"].shift(-horizon)

    labeled = (
        df.dropna(
            subset=feature_columns
            + ["Target_Return", "Target_Price", "Target_Date"]
        )
        .reset_index(drop=True)
    )

    if len(labeled) < 60:
        raise ValueError("Not enough usable rows after feature preparation.")

    X = labeled[feature_columns].astype(float)
    y = labeled["Target_Return"].astype(float)

    latest_features = df.iloc[[-1]][feature_columns].copy()

    if latest_features.isna().any(axis=None):
        raise ValueError(
            "Latest row does not have enough history to calculate indicators."
        )

    return (
        labeled,
        X,
        y,
        latest_features.astype(float),
        float(close.iloc[-1]),
    )


# --------------------------------------------------
# DATA PREPARATION: VOLATILITY MODELS
# --------------------------------------------------

def prepare_vol_data(data, horizon=5):
    """
    Target: realized volatility over the NEXT `horizon` trading days
            (annualized %), modelled in log space.
    Baseline: realized volatility over the LAST `horizon` trading days.
    """

    if horizon not in VOL_HORIZONS:
        raise ValueError(f"Volatility horizon must be one of {VOL_HORIZONS}.")

    df, base_columns, vol_columns = _build_features(data)
    feature_columns = base_columns + vol_columns

    squared = np.log(df["Close"]).diff() ** 2
    mean_squared = squared.rolling(horizon).mean()

    df["Baseline_Vol"] = np.sqrt(mean_squared) * ANNUALIZATION
    df["Target_Vol"] = np.sqrt(mean_squared.shift(-horizon)) * ANNUALIZATION
    df["Target_LogVol"] = np.log(df["Target_Vol"].clip(lower=1e-4))
    df["Target_Date"] = df["Date"].shift(-horizon)

    labeled = (
        df.dropna(
            subset=feature_columns
            + ["Baseline_Vol", "Target_LogVol", "Target_Date"]
        )
        .reset_index(drop=True)
    )

    if len(labeled) < 150:
        raise ValueError(
            "Not enough usable rows for volatility forecasting."
        )

    X = labeled[feature_columns].astype(float)
    y = labeled["Target_LogVol"].astype(float)

    latest_features = df.iloc[[-1]][feature_columns].copy()

    if latest_features.isna().any(axis=None):
        raise ValueError(
            "Latest row does not have enough history to calculate indicators."
        )

    current_vol = float(df["Baseline_Vol"].iloc[-1])

    return labeled, X, y, latest_features.astype(float), current_vol


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def directional_accuracy(previous_closes, actual, predicted):

    previous_closes = np.asarray(previous_closes, dtype=float)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.mean(
            np.sign(actual - previous_closes)
            == np.sign(predicted - previous_closes)
        ) * 100
    )


def always_up_accuracy(previous_closes, actual):
    """
    Direction benchmark: 'always predict up'. (The old 'predict no change'
    baseline scored only on days with an unchanged close.)
    """
    previous_closes = np.asarray(previous_closes, dtype=float)
    actual = np.asarray(actual, dtype=float)

    return float(np.mean(actual > previous_closes) * 100)


def prices_from_returns(previous_closes, predicted_log_returns):

    previous_closes = np.asarray(previous_closes, dtype=float)
    predicted_log_returns = np.asarray(predicted_log_returns, dtype=float)

    return previous_closes * np.exp(
        np.clip(predicted_log_returns, -0.5, 0.5)
    )


def diebold_mariano_p(baseline_errors, model_errors, horizon=1):
    """
    One-sided Diebold-Mariano test on squared errors.
    Small p => the model's errors are genuinely lower than the baseline's.
    Uses a Bartlett (Newey-West) correction for overlapping targets.
    """
    e_base = np.asarray(baseline_errors, dtype=float)
    e_model = np.asarray(model_errors, dtype=float)

    d = e_base ** 2 - e_model ** 2
    n = len(d)

    if n < 10:
        return None

    d_bar = d.mean()
    centered = d - d_bar
    variance = np.mean(centered ** 2)

    for k in range(1, horizon):
        cov = np.mean(centered[k:] * centered[:-k])
        variance += 2 * (1 - k / horizon) * cov

    if variance <= 0:
        return None

    statistic = d_bar / np.sqrt(variance / n)

    return float(1 - norm.cdf(statistic))


def _wf_split_count(n_rows, horizon, n_splits=WF_MAX_SPLITS):
    gap = max(0, horizon - 1)
    return min(n_splits, (n_rows - WF_MIN_TRAIN - gap) // WF_TEST_SIZE)


# --------------------------------------------------
# WALK-FORWARD VALIDATION (PRICE)
# --------------------------------------------------

EMPTY_WALK_FORWARD = {
    "mae": None,
    "rmse": None,
    "baseline_mae": None,
    "baseline_rmse": None,
    "directional_accuracy": None,
    "baseline_directional_accuracy": None,
    "dm_p_value": None,
    "direction_p_value": None,
    "beats_baseline": None,
    "samples": 0,
    "folds": 0,
}


def walk_forward_evaluation(
    X,
    y,
    previous_closes,
    target_prices,
    model_id,
    horizon=1,
    n_splits=WF_MAX_SPLITS,
):
    """
    Expanding-window chronological validation with fixed-size test
    windows. The gap stops training target windows from overlapping
    test target windows. Baseline is scored on exactly the same rows.
    """

    gap = max(0, horizon - 1)
    splits = _wf_split_count(len(X), horizon, n_splits)

    if splits < 2:
        return dict(EMPTY_WALK_FORWARD)

    previous_closes = np.asarray(previous_closes, dtype=float)
    target_prices = np.asarray(target_prices, dtype=float)

    actual_all, predicted_all, previous_all = [], [], []
    actual_ret_all, predicted_ret_all = [], []

    splitter = TimeSeriesSplit(
        n_splits=splits,
        test_size=WF_TEST_SIZE,
        gap=gap,
    )

    for train_idx, test_idx in splitter.split(X):

        model = create_model(model_id)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predicted_returns = np.clip(
            model.predict(X.iloc[test_idx]), -0.5, 0.5
        )

        previous = previous_closes[test_idx]

        actual_all.extend(target_prices[test_idx])
        predicted_all.extend(prices_from_returns(previous, predicted_returns))
        previous_all.extend(previous)

        actual_ret_all.extend(y.iloc[test_idx])
        predicted_ret_all.extend(predicted_returns)

    actual_all = np.asarray(actual_all)
    predicted_all = np.asarray(predicted_all)
    previous_all = np.asarray(previous_all)
    actual_ret_all = np.asarray(actual_ret_all)
    predicted_ret_all = np.asarray(predicted_ret_all)

    metrics = calculate_metrics(actual_all, predicted_all)
    baseline_metrics = calculate_metrics(actual_all, previous_all)

    dm_p = diebold_mariano_p(
        baseline_errors=actual_ret_all,
        model_errors=actual_ret_all - predicted_ret_all,
        horizon=horizon,
    )

    hits = int(np.sum(np.sign(predicted_ret_all) == np.sign(actual_ret_all)))
    n = len(actual_ret_all)
    up_share = float(np.mean(actual_ret_all > 0))

    direction_p = float(
        binomtest(
            hits, n, min(max(up_share, 0.5), 0.99), alternative="greater"
        ).pvalue
    )

    beats = bool(
        metrics["mae"] < baseline_metrics["mae"]
        and metrics["rmse"] < baseline_metrics["rmse"]
    )

    return {
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "baseline_mae": round(baseline_metrics["mae"], 4),
        "baseline_rmse": round(baseline_metrics["rmse"], 4),
        "directional_accuracy": round(
            directional_accuracy(previous_all, actual_all, predicted_all), 2
        ),
        "baseline_directional_accuracy": round(
            always_up_accuracy(previous_all, actual_all), 2
        ),
        "dm_p_value": None if dm_p is None else round(dm_p, 4),
        "direction_p_value": round(direction_p, 4),
        "beats_baseline": beats,
        "samples": int(n),
        "folds": int(splits),
    }


# --------------------------------------------------
# PRICE FORECASTS
# --------------------------------------------------

def train_model(data, model_id, horizon=1):

    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    labeled, X, y, latest_features, current_close = prepare_data(
        data, horizon=horizon
    )

    # Chronological 80/20 holdout split
    split_index = int(len(X) * 0.8)
    gap = max(0, horizon - 1)
    train_end = split_index - gap

    if train_end < 30 or len(X) - split_index < 10:
        raise ValueError(
            "Not enough data for a reliable chronological "
            "train/test split. Use more historical data "
            "or a shorter forecast horizon."
        )

    X_train, X_test = X.iloc[:train_end], X.iloc[split_index:]
    y_train = y.iloc[:train_end]

    previous_closes = (
        labeled["Close"].iloc[split_index:].to_numpy(dtype=float)
    )
    actual_prices = (
        labeled["Target_Price"].iloc[split_index:].to_numpy(dtype=float)
    )

    model = create_model(model_id)
    model.fit(X_train, y_train)

    predicted_prices = prices_from_returns(
        previous_closes, model.predict(X_test)
    )

    metrics = calculate_metrics(actual_prices, predicted_prices)

    baseline_predictions = previous_closes.copy()
    baseline_metrics = calculate_metrics(actual_prices, baseline_predictions)

    test_dates = (
        labeled["Target_Date"]
        .iloc[split_index:]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )

    model_direction = directional_accuracy(
        previous_closes, actual_prices, predicted_prices
    )
    baseline_direction = always_up_accuracy(previous_closes, actual_prices)

    walk_forward = walk_forward_evaluation(
        X,
        y,
        labeled["Close"].to_numpy(dtype=float),
        labeled["Target_Price"].to_numpy(dtype=float),
        model_id,
        horizon=horizon,
    )

    final_model = create_model(model_id)
    final_model.fit(X, y)

    next_log_return = float(final_model.predict(latest_features)[0])

    predicted_price = float(
        prices_from_returns([current_close], [next_log_return])[0]
    )

    mae_improvement = (
        (baseline_metrics["mae"] - metrics["mae"])
        / baseline_metrics["mae"] * 100
        if baseline_metrics["mae"] > 0 else None
    )

    rmse_improvement = (
        (baseline_metrics["rmse"] - metrics["rmse"])
        / baseline_metrics["rmse"] * 100
        if baseline_metrics["rmse"] > 0 else None
    )

    return {
        "horizon": horizon,
        "horizon_label": _horizon_label(horizon),
        "model_name": MODELS[model_id],
        "predicted_price": round(predicted_price, 2),
        "predicted_return_pct": round(
            (np.exp(np.clip(next_log_return, -0.5, 0.5)) - 1) * 100, 4
        ),
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "baseline_mae": round(baseline_metrics["mae"], 4),
        "baseline_rmse": round(baseline_metrics["rmse"], 4),
        "mae_improvement_pct": (
            round(float(mae_improvement), 2)
            if mae_improvement is not None else None
        ),
        "rmse_improvement_pct": (
            round(float(rmse_improvement), 2)
            if rmse_improvement is not None else None
        ),
        "test_samples": len(X_test),
        "directional_accuracy": round(model_direction, 2),
        "baseline_directional_accuracy": round(baseline_direction, 2),
        "walk_forward": walk_forward,
        "evaluation": {
            "dates": test_dates,
            "actual": [round(float(v), 4) for v in actual_prices],
            "predicted": [round(float(v), 4) for v in predicted_prices],
            "baseline": [round(float(v), 4) for v in baseline_predictions],
        },
    }


def predict_all_horizons(data, model_id):

    return {
        f"{horizon}_day": train_model(data, model_id, horizon=horizon)
        for horizon in HORIZONS
    }


def compare_models(data):
    """
    Compare all five models at every horizon (1, 5 and 10 days).
    The page picks which horizon to display, so no re-training is needed
    when the horizon dropdown changes.
    """

    results = []

    for model_id, model_name in MODELS.items():

        all_horizons = predict_all_horizons(data, model_id)
        result = all_horizons["1_day"]

        results.append({
            "model_id": model_id,
            "model_name": model_name,
            "predicted_price": result["predicted_price"],
            "predicted_return_pct": result["predicted_return_pct"],
            "mae": result["mae"],
            "rmse": result["rmse"],
            "baseline_mae": result["baseline_mae"],
            "baseline_rmse": result["baseline_rmse"],
            "mae_improvement_pct": result["mae_improvement_pct"],
            "rmse_improvement_pct": result["rmse_improvement_pct"],
            "test_samples": result["test_samples"],
            "directional_accuracy": result["directional_accuracy"],
            "baseline_directional_accuracy": (
                result["baseline_directional_accuracy"]
            ),
            "walk_forward": result["walk_forward"],
            "horizons": {
                key: {
                    "horizon": value["horizon"],
                    "horizon_label": value["horizon_label"],
                    "predicted_price": value["predicted_price"],
                    "predicted_return_pct": value["predicted_return_pct"],
                    "mae": value["mae"],
                    "rmse": value["rmse"],
                    "baseline_mae": value["baseline_mae"],
                    "baseline_rmse": value["baseline_rmse"],
                    "mae_improvement_pct": value["mae_improvement_pct"],
                    "rmse_improvement_pct": value["rmse_improvement_pct"],
                    "test_samples": value["test_samples"],
                    "directional_accuracy": value["directional_accuracy"],
                    "baseline_directional_accuracy": (
                        value["baseline_directional_accuracy"]
                    ),
                    "walk_forward": value["walk_forward"],
                }
                for key, value in all_horizons.items()
            },
        })

    def sort_key(item):
        wf = item["walk_forward"] or {}
        mae = wf.get("mae")
        return (mae if mae is not None else item["mae"], item["rmse"])

    results.sort(key=sort_key)

    return results


# --------------------------------------------------
# PREDICTION RANGES
# --------------------------------------------------

def _sigma_scale(sigma, horizon):
    """Expected size of a `horizon`-day move: daily vol x sqrt(days)."""
    return np.clip(np.asarray(sigma, dtype=float), 1e-4, None) * np.sqrt(horizon)


def _quantile_bounds(X_train, z_train, X_test):
    """Fit lower/upper quantile models; return (lower, upper) predictions."""
    lower_model = create_quantile_model(LO_Q).fit(X_train, z_train)
    upper_model = create_quantile_model(HI_Q).fit(X_train, z_train)

    lower = lower_model.predict(X_test)
    upper = upper_model.predict(X_test)

    # Guard against crossing quantiles
    return np.minimum(lower, upper), np.maximum(lower, upper)


def _interval_metrics(actual, lower, upper):

    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    inside = (actual >= lower) & (actual <= upper)
    alpha = 1 - RANGE_LEVEL

    # Interval score: width plus a penalty for every miss (lower is better)
    score = (
        (upper - lower)
        + (2 / alpha) * np.maximum(lower - actual, 0)
        + (2 / alpha) * np.maximum(actual - upper, 0)
    )

    return {
        "coverage": float(inside.mean() * 100),
        "width_pct": float(np.mean(np.exp(upper) - np.exp(lower)) * 100),
        "interval_score": float(score.mean() * 100),
    }


def _calibration_verdict(coverage):

    if coverage is None:
        return None

    gap = coverage - RANGE_LEVEL * 100

    if abs(gap) <= 3:
        return "Well calibrated"

    return "Too narrow" if gap < 0 else "Too wide"


def range_walk_forward(X, y, sigma, horizon):
    """
    Walk-forward test of both range methods. Everything (quantile models,
    historical percentiles) is fitted on the training window only.

    Both methods work on 'standardized moves' z = return / (daily vol x
    sqrt(days)), then scale back, so the range widens in volatile markets.
    """

    splits = _wf_split_count(len(X), horizon)

    if splits < 2:
        return None

    gap = max(0, horizon - 1)
    scale_all = _sigma_scale(sigma, horizon)
    z_all = y.to_numpy(dtype=float) / scale_all

    splitter = TimeSeriesSplit(
        n_splits=splits,
        test_size=WF_TEST_SIZE,
        gap=gap,
    )

    indices = []
    bounds = {"ml": ([], []), "vol_scaled": ([], [])}

    for train_idx, test_idx in splitter.split(X):

        scale_test = scale_all[test_idx]

        # Method 1: quantile gradient boosting on standardized moves
        lower_z, upper_z = _quantile_bounds(
            X.iloc[train_idx], z_all[train_idx], X.iloc[test_idx]
        )
        bounds["ml"][0].extend(lower_z * scale_test)
        bounds["ml"][1].extend(upper_z * scale_test)

        # Method 2: historical percentiles of standardized moves
        q_low, q_high = np.quantile(z_all[train_idx], [LO_Q, HI_Q])
        bounds["vol_scaled"][0].extend(q_low * scale_test)
        bounds["vol_scaled"][1].extend(q_high * scale_test)

        indices.extend(test_idx)

    return (
        np.asarray(indices),
        {
            key: (np.asarray(lo), np.asarray(hi))
            for key, (lo, hi) in bounds.items()
        },
        splits,
    )


def predict_ranges(data):
    """90% price ranges for 1, 5 and 10 trading days."""

    results = {}

    for horizon in HORIZONS:

        labeled, X, y, latest, current_close = prepare_data(
            data, horizon=horizon
        )

        sigma = labeled["Volatility_20D"].to_numpy(dtype=float)
        walk = range_walk_forward(X, y, sigma, horizon)

        # ---- Fit on all history and forecast the next range ----
        scale_all = _sigma_scale(sigma, horizon)
        z_all = y.to_numpy(dtype=float) / scale_all

        latest_scale = float(
            _sigma_scale([latest["Volatility_20D"].iloc[0]], horizon)[0]
        )

        lower_z, upper_z = _quantile_bounds(X, z_all, latest)
        q_low, q_high = np.quantile(z_all, [LO_Q, HI_Q])

        forecast_bounds = {
            "ml": (lower_z[0] * latest_scale, upper_z[0] * latest_scale),
            "vol_scaled": (q_low * latest_scale, q_high * latest_scale),
        }

        # ---- Score each method on the walk-forward windows ----
        methods = {}
        actual_returns = None

        if walk is not None:
            idx, bounds, folds = walk
            actual_returns = y.to_numpy(dtype=float)[idx]

        for key, name in METHOD_NAMES.items():

            lower_ret, upper_ret = forecast_bounds[key]

            entry = {
                "name": name,
                "lower_price": round(
                    float(current_close * np.exp(lower_ret)), 2
                ),
                "upper_price": round(
                    float(current_close * np.exp(upper_ret)), 2
                ),
                "coverage": None,
                "width_pct": None,
                "interval_score": None,
                "verdict": None,
                "samples": 0,
                "folds": 0,
            }

            if walk is not None:
                lo, hi = bounds[key]
                metrics = _interval_metrics(actual_returns, lo, hi)

                entry.update({
                    "coverage": round(metrics["coverage"], 2),
                    "width_pct": round(metrics["width_pct"], 2),
                    "interval_score": round(metrics["interval_score"], 3),
                    "verdict": _calibration_verdict(metrics["coverage"]),
                    "samples": int(len(idx)),
                    "folds": int(folds),
                })

            methods[key] = entry

        # ---- Headline method: whichever is better calibrated ----
        if walk is not None:
            headline = min(
                methods,
                key=lambda k: (
                    abs(methods[k]["coverage"] - RANGE_LEVEL * 100),
                    methods[k]["width_pct"],
                ),
            )
        else:
            headline = "vol_scaled"

        evaluation = {"dates": [], "actual": [], "lower": [], "upper": []}

        if walk is not None:
            lo, hi = bounds[headline]
            closes = labeled["Close"].to_numpy(dtype=float)[idx]

            evaluation = {
                "dates": (
                    labeled["Target_Date"].iloc[idx]
                    .dt.strftime("%Y-%m-%d").tolist()
                ),
                "actual": [
                    round(float(v), 2)
                    for v in labeled["Target_Price"].to_numpy(dtype=float)[idx]
                ],
                "lower": [round(float(v), 2) for v in closes * np.exp(lo)],
                "upper": [round(float(v), 2) for v in closes * np.exp(hi)],
            }

        results[f"{horizon}_day"] = {
            "horizon": horizon,
            "horizon_label": _horizon_label(horizon),
            "current_price": round(float(current_close), 2),
            "range_level": int(RANGE_LEVEL * 100),
            "headline_method": headline,
            "headline_name": METHOD_NAMES[headline],
            "lower_price": methods[headline]["lower_price"],
            "upper_price": methods[headline]["upper_price"],
            "width_pct": methods[headline]["width_pct"],
            "methods": methods,
            "evaluation": evaluation,
        }

    return results


# --------------------------------------------------
# VOLATILITY FORECASTS
# --------------------------------------------------

EMPTY_VOL_WALK_FORWARD = {
    "mae": None,
    "rmse": None,
    "baseline_mae": None,
    "baseline_rmse": None,
    "mae_improvement_pct": None,
    "dm_p_value": None,
    "beats_baseline": None,
    "samples": 0,
    "folds": 0,
}


def vol_walk_forward(X, y, labeled, model_id, horizon, detailed=True):
    """
    Walk-forward test of a volatility model against the baseline
    'next period's volatility = last period's volatility'.
    """

    splits = _wf_split_count(len(X), horizon)

    if splits < 2:
        return dict(EMPTY_VOL_WALK_FORWARD), None

    gap = max(0, horizon - 1)

    actual_log = y.to_numpy(dtype=float)
    baseline_vol = labeled["Baseline_Vol"].to_numpy(dtype=float)
    baseline_log = np.log(np.clip(baseline_vol, 1e-4, None))

    splitter = TimeSeriesSplit(
        n_splits=splits,
        test_size=WF_TEST_SIZE,
        gap=gap,
    )

    indices, predicted_log = [], []

    for train_idx, test_idx in splitter.split(X):

        model = create_model(model_id)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predicted_log.extend(
            np.clip(model.predict(X.iloc[test_idx]), 0.0, np.log(400.0))
        )
        indices.extend(test_idx)

    indices = np.asarray(indices)
    predicted_log = np.asarray(predicted_log)

    actual_vol = labeled["Target_Vol"].to_numpy(dtype=float)[indices]
    predicted_vol = np.exp(predicted_log)
    base_vol = baseline_vol[indices]

    metrics = calculate_metrics(actual_vol, predicted_vol)
    baseline_metrics = calculate_metrics(actual_vol, base_vol)

    dm_p = diebold_mariano_p(
        baseline_errors=actual_log[indices] - baseline_log[indices],
        model_errors=actual_log[indices] - predicted_log,
        horizon=horizon,
    )

    improvement = (
        (baseline_metrics["mae"] - metrics["mae"])
        / baseline_metrics["mae"] * 100
        if baseline_metrics["mae"] > 0 else None
    )

    summary = {
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "baseline_mae": round(baseline_metrics["mae"], 4),
        "baseline_rmse": round(baseline_metrics["rmse"], 4),
        "mae_improvement_pct": (
            None if improvement is None else round(float(improvement), 2)
        ),
        "dm_p_value": None if dm_p is None else round(dm_p, 4),
        "beats_baseline": bool(metrics["mae"] < baseline_metrics["mae"]),
        "samples": int(len(indices)),
        "folds": int(splits),
    }

    evaluation = None

    if detailed:
        evaluation = {
            "dates": (
                labeled["Target_Date"].iloc[indices]
                .dt.strftime("%Y-%m-%d").tolist()
            ),
            "actual": [round(float(v), 3) for v in actual_vol],
            "predicted": [round(float(v), 3) for v in predicted_vol],
            "baseline": [round(float(v), 3) for v in base_vol],
        }

    return summary, evaluation


def volatility_forecast(data, model_id, horizon=5, detailed=True):

    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    labeled, X, y, latest, current_vol = prepare_vol_data(
        data, horizon=horizon
    )

    summary, evaluation = vol_walk_forward(
        X, y, labeled, model_id, horizon, detailed=detailed
    )

    final_model = create_model(model_id)
    final_model.fit(X, y)

    predicted_log = float(
        np.clip(final_model.predict(latest)[0], 0.0, np.log(400.0))
    )
    forecast_vol = float(np.exp(predicted_log))

    return {
        "horizon": horizon,
        "horizon_label": f"next {horizon} trading days",
        "model_name": MODELS[model_id],
        "forecast_vol": round(forecast_vol, 2),
        "current_vol": round(current_vol, 2),
        "change_pct": round((forecast_vol / current_vol - 1) * 100, 2)
        if current_vol > 0 else None,
        "walk_forward": summary,
        "evaluation": evaluation,
    }


def predict_volatility_all(data, model_id):

    return {
        f"{horizon}_day": volatility_forecast(
            data, model_id, horizon=horizon, detailed=True
        )
        for horizon in VOL_HORIZONS
    }


def compare_volatility_models(data):

    results = []

    for model_id, model_name in MODELS.items():

        horizons = {}

        for horizon in VOL_HORIZONS:
            result = volatility_forecast(
                data, model_id, horizon=horizon, detailed=False
            )
            horizons[f"{horizon}_day"] = {
                "horizon": horizon,
                "forecast_vol": result["forecast_vol"],
                "current_vol": result["current_vol"],
                "walk_forward": result["walk_forward"],
            }

        results.append({
            "model_id": model_id,
            "model_name": model_name,
            "horizons": horizons,
        })

    return results