
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_model(company_data):

    company_data = company_data.sort_values("Date").copy()

    company_data["Previous_Close"] = (
        company_data["Close"].shift(1)
    )

    company_data["Target"] = (
        company_data["Close"].shift(-1)
    )

    company_data = company_data.dropna()

    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Previous_Close"
    ]

    X = company_data[features]
    y = company_data["Target"]

    if len(X) < 5:
        raise ValueError("Not enough data")

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    latest_data = company_data.iloc[-1]

    latest_features = latest_data[
        features
    ].to_frame().T

    predicted_price = model.predict(
        latest_features
    )[0]

    return {
        "predicted_price": round(
            float(predicted_price), 2
        ),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2)
    }
