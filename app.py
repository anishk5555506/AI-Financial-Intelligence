
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf

from model import MODELS, train_model, compare_models

app = Flask(__name__)

TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA"
}

TRADING_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.04


def get_company_data(company):
    ticker = TICKERS.get(company)

    if ticker is None:
        return None

    try:
        data = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        required = ["Date", "Open", "High", "Low", "Close", "Volume"]

        if not all(column in data.columns for column in required):
            return None

        data = data[required].copy()
        data["Date"] = pd.to_datetime(data["Date"])

        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        data[numeric_columns] = data[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        data = data.dropna(subset=numeric_columns)
        data = data[data["Close"] > 0]
        data = data.sort_values("Date").reset_index(drop=True)

        return data

    except Exception:
        app.logger.exception("Failed to download data for %s", company)
        return None


def json_safe(value):
    """Convert NumPy values to values Flask can serialize."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/companies")
def get_companies():
    return jsonify(list(TICKERS.keys()))


@app.route("/api/models")
def get_models():
    return jsonify([
        {"id": model_id, "name": model_name}
        for model_id, model_name in MODELS.items()
    ])


@app.route("/api/company/<company>")
def get_company_data_api(company):
    if company not in TICKERS:
        return jsonify({"error": "Company not found."}), 404

    data = get_company_data(company)

    if data is None or data.empty:
        return jsonify({
            "error": "Unable to retrieve historical data from Yahoo Finance."
        }), 502

    if len(data) < 2:
        return jsonify({"error": "Not enough historical data."}), 400

    latest = data.iloc[-1]
    previous = data.iloc[-2]

    change = float(latest["Close"] - previous["Close"])
    percentage_change = (
        change / float(previous["Close"]) * 100
        if previous["Close"] != 0 else 0.0
    )

    trend = (
        "Positive" if percentage_change > 0
        else "Negative" if percentage_change < 0
        else "Neutral"
    )

    return jsonify({
        "company": company,
        "latest_price": round(float(latest["Close"]), 2),
        "previous_price": round(float(previous["Close"]), 2),
        "price_change": round(change, 2),
        "percentage_change": round(percentage_change, 2),
        "volume": int(latest["Volume"]),
        "trend": trend,
        "dates": data["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "prices": data["Close"].astype(float).tolist()
    })


@app.route("/api/risk/<company>")
def calculate_risk(company):
    if company not in TICKERS:
        return jsonify({"error": "Company not found."}), 404

    data = get_company_data(company)

    if data is None or len(data) < 30:
        return jsonify({
            "error": "Unable to retrieve enough historical data for risk analysis."
        }), 502

    close = data["Close"].astype(float)
    daily_returns = close.pct_change().dropna()

    if len(daily_returns) < 2:
        return jsonify({
            "error": "Not enough return observations for risk analysis."
        }), 400

    cumulative_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    annualized_volatility = daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100

    running_peak = close.cummax()
    drawdowns = (close / running_peak - 1) * 100
    maximum_drawdown = drawdowns.min()

    daily_risk_free = (
        (1 + DEFAULT_RISK_FREE_RATE) ** (1 / TRADING_DAYS)
    ) - 1

    daily_volatility = daily_returns.std(ddof=1)

    if daily_volatility > 0:
        sharpe_ratio = (
            (daily_returns.mean() - daily_risk_free)
            / daily_volatility
        ) * np.sqrt(TRADING_DAYS)
    else:
        sharpe_ratio = 0.0

    cumulative_returns = (close / close.iloc[0] - 1) * 100

    return jsonify(json_safe({
        "company": company,
        "ticker": TICKERS[company],
        "start_date": data["Date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": data["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "observations": len(data),
        "risk_free_rate_pct": DEFAULT_RISK_FREE_RATE * 100,
        "cumulative_return_pct": round(float(cumulative_return), 2),
        "annualized_volatility_pct": round(float(annualized_volatility), 2),
        "maximum_drawdown_pct": round(float(maximum_drawdown), 2),
        "sharpe_ratio": round(float(sharpe_ratio), 3),
        "dates": data["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "cumulative_returns": [
            round(float(value), 4) for value in cumulative_returns
        ],
        "drawdowns": [
            round(float(value), 4) for value in drawdowns
        ],
        "note": (
            "Historical metrics calculated from adjusted daily closing prices. "
            "They do not guarantee future results."
        )
    }))


@app.route("/api/predict/<company>")
def predict_company(company):
    if company not in TICKERS:
        return jsonify({"error": "Company not found."}), 404

    model_id = request.args.get("model", "random_forest")

    if model_id not in MODELS:
        return jsonify({"error": "Unknown model selected."}), 400

    data = get_company_data(company)

    if data is None or len(data) < 100:
        return jsonify({
            "error": "Unable to retrieve enough historical data."
        }), 502

    try:
        result = train_model(data, model_id)

        return jsonify(json_safe({
            "company": company,
            "model_id": model_id,
            "model_name": result["model_name"],
            "current_price": round(float(data.iloc[-1]["Close"]), 2),
            "predicted_price": result["predicted_price"],
            "mae": result["mae"],
            "rmse": result["rmse"],
            "baseline_mae": result["baseline_mae"],
            "baseline_rmse": result["baseline_rmse"],
            "test_samples": result["test_samples"],
            "directional_accuracy": result["directional_accuracy"],
            "baseline_directional_accuracy": result[
                "baseline_directional_accuracy"
            ],
            "walk_forward": result["walk_forward"],
            "evaluation": result["evaluation"]
        }))

    except Exception:
        app.logger.exception(
            "Prediction failed for %s using %s", company, model_id
        )
        return jsonify({
            "error": "Prediction failed. Check the VS Code terminal."
        }), 500


@app.route("/api/compare/<company>")
def compare_company_models(company):
    if company not in TICKERS:
        return jsonify({"error": "Company not found."}), 404

    data = get_company_data(company)

    if data is None or len(data) < 100:
        return jsonify({
            "error": "Unable to retrieve enough historical data."
        }), 502

    try:
        results = compare_models(data)
        return jsonify(json_safe({
            "company": company,
            "results": results
        }))

    except Exception:
        app.logger.exception("Model comparison failed for %s", company)
        return jsonify({
            "error": "Model comparison failed. Check the VS Code terminal."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
