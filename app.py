from flask import Flask, render_template, jsonify
from model import train_model
import pandas as pd

app = Flask(__name__)

# Load financial data
data = pd.read_csv("data/financial_data.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/companies")
def get_companies():
    companies = data["Company"].unique().tolist()

    return jsonify(companies)


@app.route("/api/company/<company>")
def get_company_data(company):

    company_data = data[data["Company"] == company]

    if company_data.empty:
        return jsonify({"error": "Company not found"}), 404

    company_data = company_data.sort_values("Date")

    latest = company_data.iloc[-1]
    previous = company_data.iloc[-2]

    price_change = latest["Close"] - previous["Close"]
    percentage_change = (price_change / previous["Close"]) * 100

    if percentage_change > 0:
        trend = "Positive"
    elif percentage_change < 0:
        trend = "Negative"
    else:
        trend = "Neutral"

    result = {
        "company": company,
        "latest_price": float(latest["Close"]),
        "previous_price": float(previous["Close"]),
        "price_change": round(float(price_change), 2),
        "percentage_change": round(float(percentage_change), 2),
        "volume": int(latest["Volume"]),
        "trend": trend,
        "dates": company_data["Date"].tolist(),
        "prices": company_data["Close"].tolist()
    }

    return jsonify(result)


@app.route("/api/predict/<company>")
def predict_company(company):

    company_data = data[
        data["Company"] == company
    ].copy()

    if company_data.empty:
        return jsonify({
            "error": "Company not found"
        }), 404

    company_data["Date"] = pd.to_datetime(
        company_data["Date"]
    )

    company_data = company_data.sort_values("Date")

    if len(company_data) < 10:
        return jsonify({
            "error": "Not enough historical data"
        }), 400

    try:
        result = train_model(company_data)

        return jsonify({
    "company": company,
    "current_price": float(company_data.iloc[-1]["Close"]),
    "predicted_price": result["predicted_price"],
    "mae": result["mae"],
    "rmse": result["rmse"]
})

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)