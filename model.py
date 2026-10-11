
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


MODELS = {
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "linear_regression": "Linear Regression",
    "knn": "KNN Regressor",
    "svr": "Support Vector Regression"
}

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Previous_Close",
    "Lag_2",
    "Lag_3",
    "Lag_5",
    "Lag_10",
    "Return_1D",
    "Return_5D",
    "MA_5",
    "MA_10",
    "MA_20"
]


def create_model(model_id):
    if model_id == "random_forest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42
        )

    if model_id == "linear_regression":
        return make_pipeline(
            StandardScaler(),
            LinearRegression()
        )

    if model_id == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=5,
                weights="distance"
            )
        )

    if model_id == "svr":
        return make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.1
            )
        )

    raise ValueError(f"Unknown model: {model_id}")


def prepare_data(data):
    if data is None or data.empty:
        raise ValueError("No historical data was supplied.")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns)
    df = df[df["Close"] > 0].reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("At least 100 historical rows are required.")

    df["Previous_Close"] = df["Close"]
    df["Lag_2"] = df["Close"].shift(1)
    df["Lag_3"] = df["Close"].shift(2)
    df["Lag_5"] = df["Close"].shift(4)
    df["Lag_10"] = df["Close"].shift(9)

    df["Return_1D"] = df["Close"].pct_change()
    df["Return_5D"] = df["Close"].pct_change(5)

    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()

    # Each feature row at date t predicts the close at date t+1.
    df["Target"] = df["Close"].shift(-1)
    df["Target_Date"] = df["Date"].shift(-1)

    # Keep the final feature row for the next-session estimate.
    next_features = df.iloc[[-1]][FEATURE_COLUMNS].copy()

    # Historical training rows must have valid features and a target.
    labeled = df.dropna(
        subset=FEATURE_COLUMNS + ["Target", "Target_Date"]
    ).copy()

    labeled = labeled.reset_index(drop=True)

    if len(labeled) < 50:
        raise ValueError("Not enough usable rows after feature preparation.")

    X = labeled[FEATURE_COLUMNS].astype(float)
    y = labeled["Target"].astype(float)

    return labeled, X, y, next_features.astype(float)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))
    }


def directional_accuracy(previous_close, actual, predicted):
    previous_close = np.asarray(previous_close, dtype=float)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    actual_direction = np.sign(actual - previous_close)
    predicted_direction = np.sign(predicted - previous_close)

    return float(np.mean(actual_direction == predicted_direction) * 100)


def walk_forward_evaluation(X, y, previous_closes, model_id, n_splits=5):
    """
    Expanding-window validation. Each fold trains only on observations
    earlier than its test observations.
    """
    splits = min(n_splits, len(X) // 20)

    if splits < 2:
        return {
            "mae": None,
            "rmse": None,
            "directional_accuracy": None,
            "samples": 0,
            "folds": 0
        }

    splitter = TimeSeriesSplit(n_splits=splits)

    all_actual = []
    all_predicted = []
    all_previous = []

    for train_indices, test_indices in splitter.split(X):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        model = create_model(model_id)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        all_actual.extend(y_test.to_numpy())
        all_predicted.extend(predictions)
        all_previous.extend(previous_closes.iloc[test_indices].to_numpy())

    metrics = calculate_metrics(all_actual, all_predicted)

    return {
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "directional_accuracy": round(
            directional_accuracy(all_previous, all_actual, all_predicted), 2
        ),
        "samples": len(all_actual),
        "folds": splits
    }


def train_model(data, model_id):
    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    labeled, X, y, next_features = prepare_data(data)

    # Keep the final 20% for a chronological holdout test.
    split_index = int(len(X) * 0.8)

    if split_index < 30 or len(X) - split_index < 10:
        raise ValueError("Not enough data for a reliable train/test split.")

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = create_model(model_id)
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    previous_closes = X_test["Close"].to_numpy()
    actual_values = y_test.to_numpy()

    model_metrics = calculate_metrics(actual_values, test_predictions)

    # Persistence baseline: next close is estimated as the current close.
    baseline_predictions = previous_closes
    baseline_metrics = calculate_metrics(actual_values, baseline_predictions)

    test_dates = (
        labeled["Target_Date"]
        .iloc[split_index:]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )

    holdout_directional_accuracy = directional_accuracy(
        previous_closes,
        actual_values,
        test_predictions
    )

    baseline_directional_accuracy = directional_accuracy(
        previous_closes,
        actual_values,
        baseline_predictions
    )

    # Run expanding-window validation on all labeled rows.
    walk_forward = walk_forward_evaluation(
        X=X,
        y=y,
        previous_closes=labeled["Close"],
        model_id=model_id,
        n_splits=5
    )

    # Refit on all labeled historical data for the next-session estimate.
    final_model = create_model(model_id)
    final_model.fit(X, y)

    next_prediction = float(final_model.predict(next_features)[0])

    return {
        "model_name": MODELS[model_id],
        "predicted_price": round(next_prediction, 2),
        "mae": round(model_metrics["mae"], 4),
        "rmse": round(model_metrics["rmse"], 4),
        "baseline_mae": round(baseline_metrics["mae"], 4),
        "baseline_rmse": round(baseline_metrics["rmse"], 4),
        "test_samples": len(X_test),
        "directional_accuracy": round(holdout_directional_accuracy, 2),
        "baseline_directional_accuracy": round(
            baseline_directional_accuracy, 2
        ),
        "walk_forward": walk_forward,
        "evaluation": {
            "dates": test_dates,
            "actual": [round(float(value), 4) for value in actual_values],
            "predicted": [
                round(float(value), 4) for value in test_predictions
            ],
            "baseline": [
                round(float(value), 4) for value in baseline_predictions
            ]
        }
    }


def compare_models(data):
    results = []

    for model_id, model_name in MODELS.items():
        result = train_model(data, model_id)

        results.append({
            "model_id": model_id,
            "model_name": model_name,
            "predicted_price": result["predicted_price"],
            "mae": result["mae"],
            "rmse": result["rmse"],
            "baseline_mae": result["baseline_mae"],
            "baseline_rmse": result["baseline_rmse"],
            "test_samples": result["test_samples"],
            "directional_accuracy": result["directional_accuracy"],
            "baseline_directional_accuracy": (
                result["baseline_directional_accuracy"]
            ),
            "walk_forward": result["walk_forward"]
        })

    results.sort(key=lambda result: (result["mae"], result["rmse"]))

    return results
