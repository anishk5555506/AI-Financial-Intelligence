import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


MODELS = {
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "linear_regression": "Linear Regression",
    "knn": "KNN Regressor",
    "svr": "Support Vector Regression",
}

HORIZONS = (1, 5, 10)

FEATURE_COLUMNS = [
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


def create_model(model_id):
    if model_id == "random_forest":
        return RandomForestRegressor(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )

    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=10,
            l2_regularization=2.0,
            learning_rate=0.05,
            random_state=42,
        )

    if model_id == "linear_regression":
        return make_pipeline(
            StandardScaler(),
            LinearRegression(),
        )

    if model_id == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=7,
                weights="distance",
            ),
        )

    if model_id == "svr":
        return make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.001,
            ),
        )

    raise ValueError(f"Unknown model: {model_id}")


def _rsi(close, period=14):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))

    result = result.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100,
    )

    result = result.mask(
        (avg_loss == 0) & (avg_gain == 0),
        50,
    )

    return result


def prepare_data(data, horizon=1):
    """Build historical indicators and a future log-return target."""

    if horizon not in HORIZONS:
        raise ValueError(f"Horizon must be one of {HORIZONS}.")

    if data is None or data.empty:
        raise ValueError("No historical data was supplied.")

    df = data.copy()

    required = [
        "Date", "Open", "High", "Low", "Close", "Volume"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    df = df[
        (df["Close"] > 0) & (df["Open"] > 0)
    ].reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            "At least 100 historical rows are required."
        )

    close = df["Close"]
    volume = df["Volume"].clip(lower=0)
    returns = close.pct_change()

    df["Return_1D"] = close.pct_change(1)

    for period in [2, 3, 5, 10, 20]:
        df[f"Return_{period}D"] = close.pct_change(period)

    for lag in [1, 2, 3, 5, 10]:
        df[f"Return_Lag_{lag}"] = returns.shift(lag)

    for window in [5, 10, 20, 50]:
        moving_average = close.rolling(window).mean()
        df[f"Close_vs_MA{window}"] = (
            close / moving_average - 1
        )

    for window in [5, 10, 20]:
        df[f"Volatility_{window}D"] = (
            returns.rolling(window).std()
        )

    df["RSI_14"] = _rsi(close) / 100.0

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = (ema12 - ema26) / close

    macd_signal = (
        (ema12 - ema26)
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["MACD_Signal"] = macd_signal / close

    df["MACD_Histogram"] = (
        ema12 - ema26 - macd_signal
    ) / close

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20

    band_width = (upper_band - lower_band) / ma20

    df["BB_Position"] = (
        (close - lower_band)
        / (upper_band - lower_band).replace(0, np.nan)
    )

    df["BB_Width"] = band_width

    previous_close = close.shift(1)

    df["Intraday_Range"] = (
        df["High"] - df["Low"]
    ) / close

    df["Open_Gap"] = (
        df["Open"] / previous_close - 1
    )

    df["Close_vs_Open"] = (
        close / df["Open"] - 1
    )

    df["Volume_Change"] = (
        volume.pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )

    for window in [5, 20]:
        volume_ma = volume.rolling(window).mean()

        df[f"Volume_vs_MA{window}"] = (
            volume / volume_ma.replace(0, np.nan) - 1
        )

    rolling_high = df["High"].rolling(20).max()
    rolling_low = df["Low"].rolling(20).min()

    df["High_vs_20D"] = (
        close / rolling_high - 1
    )

    df["Low_vs_20D"] = (
        close / rolling_low - 1
    )

    df["Target_Return"] = np.log(
        close.shift(-horizon) / close
    )

    df["Target_Price"] = close.shift(-horizon)
    df["Target_Date"] = df["Date"].shift(-horizon)

    labeled = (
        df.dropna(
            subset=FEATURE_COLUMNS + [
                "Target_Return",
                "Target_Price",
                "Target_Date",
            ]
        )
        .reset_index(drop=True)
    )

    if len(labeled) < 60:
        raise ValueError(
            "Not enough usable rows after feature preparation."
        )

    X = labeled[FEATURE_COLUMNS].astype(float)
    y = labeled["Target_Return"].astype(float)

    if (
        not np.isfinite(X.to_numpy()).all()
        or not np.isfinite(y.to_numpy()).all()
    ):
        raise ValueError(
            "Non-finite values remain after feature preparation."
        )

    latest_features = df.iloc[[-1]][FEATURE_COLUMNS].copy()

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


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
    }


def directional_accuracy(
    previous_closes,
    actual,
    predicted,
):
    previous_closes = np.asarray(
        previous_closes,
        dtype=float,
    )

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.mean(
            np.sign(actual - previous_closes)
            == np.sign(predicted - previous_closes)
        ) * 100
    )


def prices_from_returns(
    previous_closes,
    predicted_log_returns,
):
    previous_closes = np.asarray(
        previous_closes,
        dtype=float,
    )

    predicted_log_returns = np.asarray(
        predicted_log_returns,
        dtype=float,
    )

    return previous_closes * np.exp(
        np.clip(predicted_log_returns, -0.5, 0.5)
    )


def walk_forward_evaluation(
    X,
    y,
    previous_closes,
    target_prices,
    model_id,
    n_splits=5,
):
    splits = min(n_splits, len(X) // 20)

    if splits < 2:
        return {
            "mae": None,
            "rmse": None,
            "baseline_mae": None,
            "baseline_rmse": None,
            "directional_accuracy": None,
            "baseline_directional_accuracy": None,
            "samples": 0,
            "folds": 0,
        }

    previous_closes = np.asarray(
        previous_closes,
        dtype=float,
    )

    target_prices = np.asarray(
        target_prices,
        dtype=float,
    )

    actual_all = []
    predicted_all = []
    baseline_all = []
    previous_all = []

    for train_idx, test_idx in TimeSeriesSplit(
        n_splits=splits
    ).split(X):

        model = create_model(model_id)

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        predicted_returns = model.predict(
            X.iloc[test_idx]
        )

        previous = previous_closes[test_idx]

        predicted = prices_from_returns(
            previous,
            predicted_returns,
        )

        actual_all.extend(target_prices[test_idx])
        predicted_all.extend(predicted)
        baseline_all.extend(previous)
        previous_all.extend(previous)

    metrics = calculate_metrics(
        actual_all,
        predicted_all,
    )

    baseline_metrics = calculate_metrics(
        actual_all,
        baseline_all,
    )

    return {
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "baseline_mae": round(
            baseline_metrics["mae"], 4
        ),
        "baseline_rmse": round(
            baseline_metrics["rmse"], 4
        ),
        "directional_accuracy": round(
            directional_accuracy(
                previous_all,
                actual_all,
                predicted_all,
            ),
            2,
        ),
        "baseline_directional_accuracy": round(
            directional_accuracy(
                previous_all,
                actual_all,
                baseline_all,
            ),
            2,
        ),
        "samples": len(actual_all),
        "folds": splits,
    }


def train_model(data, model_id, horizon=1):
    if model_id not in MODELS:
        raise ValueError(
            f"Unknown model: {model_id}"
        )

    (
        labeled,
        X,
        y,
        latest_features,
        current_close,
    ) = prepare_data(
        data,
        horizon=horizon,
    )

    split_index = int(len(X) * 0.8)

    if split_index < 30 or len(X) - split_index < 10:
        raise ValueError(
            "Not enough data for a reliable chronological train/test split."
        )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    previous_closes = (
        labeled["Close"]
        .iloc[split_index:]
        .to_numpy(dtype=float)
    )

    actual_prices = (
        labeled["Target_Price"]
        .iloc[split_index:]
        .to_numpy(dtype=float)
    )

    model = create_model(model_id)
    model.fit(X_train, y_train)

    predicted_returns = model.predict(X_test)

    predicted_prices = prices_from_returns(
        previous_closes,
        predicted_returns,
    )

    metrics = calculate_metrics(
        actual_prices,
        predicted_prices,
    )

    baseline_predictions = previous_closes.copy()

    baseline_metrics = calculate_metrics(
        actual_prices,
        baseline_predictions,
    )

    test_dates = (
        labeled["Target_Date"]
        .iloc[split_index:]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )

    model_direction = directional_accuracy(
        previous_closes,
        actual_prices,
        predicted_prices,
    )

    baseline_direction = directional_accuracy(
        previous_closes,
        actual_prices,
        baseline_predictions,
    )

    walk_forward = walk_forward_evaluation(
        X,
        y,
        labeled["Close"].to_numpy(dtype=float),
        labeled["Target_Price"].to_numpy(dtype=float),
        model_id,
        n_splits=5,
    )

    final_model = create_model(model_id)
    final_model.fit(X, y)

    next_log_return = float(
        final_model.predict(latest_features)[0]
    )

    predicted_price = float(
        prices_from_returns(
            [current_close],
            [next_log_return],
        )[0]
    )

    mae_improvement = (
        (
            baseline_metrics["mae"] - metrics["mae"]
        ) / baseline_metrics["mae"] * 100
        if baseline_metrics["mae"] > 0
        else None
    )

    rmse_improvement = (
        (
            baseline_metrics["rmse"] - metrics["rmse"]
        ) / baseline_metrics["rmse"] * 100
        if baseline_metrics["rmse"] > 0
        else None
    )

    return {
        "horizon": horizon,
        "horizon_label": (
            "next trading day"
            if horizon == 1
            else f"{horizon} trading days"
        ),
        "model_name": MODELS[model_id],
        "predicted_price": round(
            predicted_price, 2
        ),
        "predicted_return_pct": round(
            (
                np.exp(
                    np.clip(
                        next_log_return,
                        -0.5,
                        0.5,
                    )
                ) - 1
            ) * 100,
            4,
        ),
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "baseline_mae": round(
            baseline_metrics["mae"], 4
        ),
        "baseline_rmse": round(
            baseline_metrics["rmse"], 4
        ),
        "mae_improvement_pct": (
            round(float(mae_improvement), 2)
            if mae_improvement is not None
            else None
        ),
        "rmse_improvement_pct": (
            round(float(rmse_improvement), 2)
            if rmse_improvement is not None
            else None
        ),
        "test_samples": len(X_test),
        "directional_accuracy": round(
            model_direction, 2
        ),
        "baseline_directional_accuracy": round(
            baseline_direction, 2
        ),
        "walk_forward": walk_forward,
        "evaluation": {
            "dates": test_dates,
            "actual": [
                round(float(value), 4)
                for value in actual_prices
            ],
            "predicted": [
                round(float(value), 4)
                for value in predicted_prices
            ],
            "baseline": [
                round(float(value), 4)
                for value in baseline_predictions
            ],
        },
    }


def predict_all_horizons(data, model_id):
    """Return independent horizon-specific models and evaluations."""
    results = {}

    for horizon in HORIZONS:
        result = train_model(
            data,
            model_id,
            horizon=horizon,
        )

        results[f"{horizon}_day"] = result

    return results


def compare_models(data):
    results = []

    for model_id, model_name in MODELS.items():
        result = train_model(
            data,
            model_id,
            horizon=1,
        )

        all_horizons = predict_all_horizons(
            data,
            model_id,
        )

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
                    "predicted_price": value["predicted_price"],
                    "predicted_return_pct": value["predicted_return_pct"],
                    "mae": value["mae"],
                    "rmse": value["rmse"],
                    "baseline_mae": value["baseline_mae"],
                    "baseline_rmse": value["baseline_rmse"],
                    "directional_accuracy": value["directional_accuracy"],
                    "walk_forward": value["walk_forward"],
                }
                for key, value in all_horizons.items()
            },
        })

    results.sort(
        key=lambda item: (
            item["mae"],
            item["rmse"],
        )
    )

    return results