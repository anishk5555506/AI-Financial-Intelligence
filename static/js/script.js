"use strict";

let priceChart = null;
let predictionChart = null;
let comparisonChart = null;
let cumulativeReturnChart = null;
let drawdownChart = null;

let currentCompany = "Apple";
let currentModel = "random_forest";

const $ = (id) => document.getElementById(id);

const numberFormat = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2
});

function setText(id, value) {
    const element = $(id);
    if (element) {
        element.textContent = value;
    }
}

function formatPrice(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return "—";
    }
    return "$" + number.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function formatNumber(value, digits = 2) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return "—";
    }
    return number.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
    });
}

function formatPercent(value, digits = 2) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return "—";
    }
    return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}%`;
}

function formatMetric(value, digits = 2) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "—";
    }
    return numberFormat.format(Number(value));
}

function setMetricColor(id, value) {
    const element = $(id);
    if (!element) {
        return;
    }

    element.classList.remove("positive", "negative", "neutral");

    const number = Number(value);
    if (!Number.isFinite(number)) {
        element.classList.add("neutral");
    } else if (number > 0) {
        element.classList.add("positive");
    } else if (number < 0) {
        element.classList.add("negative");
    } else {
        element.classList.add("neutral");
    }
}

function setStatus(id, message, type = "") {
    const element = $(id);
    if (!element) {
        return;
    }

    element.textContent = message;
    element.classList.remove("success", "error", "loading");

    if (type) {
        element.classList.add(type);
    }
}

async function fetchJSON(url) {
    const response = await fetch(url, {
        headers: { "Accept": "application/json" }
    });

    let payload;
    try {
        payload = await response.json();
    } catch {
        throw new Error(`The server returned an invalid response (${response.status}).`);
    }

    if (!response.ok) {
        throw new Error(payload.error || `Request failed (${response.status}).`);
    }

    return payload;
}

function destroyChart(chart) {
    if (chart) {
        chart.destroy();
    }
}

function commonChartOptions(yTitle = "") {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: "index",
            intersect: false
        },
        plugins: {
            legend: {
                display: true,
                position: "top"
            },
            tooltip: {
                enabled: true
            }
        },
        scales: {
            x: {
                ticks: {
                    maxTicksLimit: 10,
                    maxRotation: 0
                },
                grid: {
                    display: false
                }
            },
            y: {
                title: {
                    display: Boolean(yTitle),
                    text: yTitle
                },
                beginAtZero: false
            }
        }
    };
}

function makeLineChart(canvasId, labels, datasets, yTitle = "") {
    const canvas = $(canvasId);
    if (!canvas) {
        return null;
    }

    return new Chart(canvas, {
        type: "line",
        data: {
            labels,
            datasets
        },
        options: commonChartOptions(yTitle)
    });
}

function makeBarChart(canvasId, labels, datasets, yTitle = "") {
    const canvas = $(canvasId);
    if (!canvas) {
        return null;
    }

    return new Chart(canvas, {
        type: "bar",
        data: {
            labels,
            datasets
        },
        options: {
            ...commonChartOptions(yTitle),
            scales: {
                ...commonChartOptions(yTitle).scales,
                y: {
                    ...commonChartOptions(yTitle).scales.y,
                    beginAtZero: true
                }
            }
        }
    });
}

async function loadCompanyData() {
    currentCompany = $("companySelect").value;

    setStatus("globalStatus", `Loading ${currentCompany} market data...`, "loading");

    try {
        const data = await fetchJSON(
            `/api/company/${encodeURIComponent(currentCompany)}`
        );

        setText("latestPrice", formatPrice(data.latest_price));
        setText("dailyChange", formatPrice(data.price_change));
        setText("dailyChangePct", formatPercent(data.percentage_change));
        setText("tradingVolume", numberFormat.format(data.volume));
        setText("marketTrend", data.trend);

        setMetricColor("dailyChange", data.price_change);
        setMetricColor("dailyChangePct", data.percentage_change);
        setMetricColor(
            "marketTrend",
            data.trend === "Positive" ? 1 :
                data.trend === "Negative" ? -1 : 0
        );

        setText(
            "lastUpdated",
            `Latest available date: ${data.dates[data.dates.length - 1] || "—"}`
        );

        destroyChart(priceChart);
        priceChart = makeLineChart(
            "priceChart",
            data.dates,
            [{
                label: `${currentCompany} closing price`,
                data: data.prices,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.15
            }],
            "Price (USD)"
        );

        setStatus("globalStatus", `${currentCompany} data loaded successfully.`, "success");

        await Promise.allSettled([
            loadRiskData(),
            loadPrediction()
        ]);
    } catch (error) {
        setStatus("globalStatus", error.message, "error");
    }
}

function renderHorizonForecasts(horizons) {
    const horizonCards = [
        {
            key: "1_day",
            priceId: "forecast1Price",
            returnId: "forecast1Return",
            dateId: "forecast1Date",
            fallbackDate: "Next trading day estimate"
        },
        {
            key: "5_day",
            priceId: "forecast5Price",
            returnId: "forecast5Return",
            fallbackDate: "Five trading days ahead"
        },
        {
            key: "10_day",
            priceId: "forecast10Price",
            returnId: "forecast10Return",
            fallbackDate: "Ten trading days ahead"
        }
    ];

    horizonCards.forEach((card) => {
        const result = horizons && horizons[card.key];

        if (!result) {
            setText(card.priceId, "—");
            setText(card.returnId, "—");
            if (card.dateId) {
                setText(card.dateId, card.fallbackDate);
            }
            return;
        }

        setText(card.priceId, formatPrice(result.predicted_price));
        setText(
            card.returnId,
            `${formatPercent(result.predicted_return_pct)} estimated return`
        );

        setMetricColor(card.returnId, result.predicted_return_pct);

        if (card.dateId) {
            setText(card.dateId, result.horizon_label || card.fallbackDate);
        }
    });
}

function renderPredictionEvaluation(data) {
    setText("modelMAE", formatMetric(data.mae));
    setText("modelRMSE", formatMetric(data.rmse));
    setText("baselineMAE", formatMetric(data.baseline_mae));
    setText("baselineRMSE", formatMetric(data.baseline_rmse));

    setText(
        "directionalAccuracy",
        data.directional_accuracy == null
            ? "—"
            : `${formatMetric(data.directional_accuracy)}%`
    );

    setText(
        "baselineDirectionalAccuracy",
        data.baseline_directional_accuracy == null
            ? "—"
            : `${formatMetric(data.baseline_directional_accuracy)}%`
    );

    const walkForward = data.walk_forward || {};

    setText("walkForwardMAE", formatMetric(walkForward.mae));
    setText("walkForwardRMSE", formatMetric(walkForward.rmse));

    const evaluation = data.evaluation || {};
    const labels = evaluation.dates || [];

    destroyChart(predictionChart);

    if (labels.length > 0) {
        predictionChart = makeLineChart(
            "predictionChart",
            labels,
            [
                {
                    label: "Actual price",
                    data: evaluation.actual || [],
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.05
                },
                {
                    label: "Model prediction",
                    data: evaluation.predicted || [],
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.05
                },
                {
                    label: "Previous-close baseline",
                    data: evaluation.baseline || [],
                    borderWidth: 1.5,
                    pointRadius: 0,
                    borderDash: [5, 5],
                    tension: 0
                }
            ],
            "Price (USD)"
        );
    }
}

async function loadPrediction() {
    currentCompany = $("companySelect").value;
    currentModel = $("modelSelect").value;

    setStatus("predictionStatus", "Training models and generating forecasts...", "loading");

    if ($("predictButton")) {
        $("predictButton").disabled = true;
    }

    try {
        const query = new URLSearchParams({
            model: currentModel,
            horizon: "all"
        });

        const data = await fetchJSON(
            `/api/predict/${encodeURIComponent(currentCompany)}?${query.toString()}`
        );

        renderHorizonForecasts(data.horizons);
        renderPredictionEvaluation(data);

        setStatus(
            "predictionStatus",
            `${data.model_name} forecasts generated for ${currentCompany}.`,
            "success"
        );
    } catch (error) {
        setStatus("predictionStatus", error.message, "error");
    } finally {
        if ($("predictButton")) {
            $("predictButton").disabled = false;
        }
    }
}

function renderComparisonTable(results) {
    const body = $("comparisonTableBody");
    if (!body) {
        return;
    }

    body.replaceChildren();

    if (!Array.isArray(results) || results.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 6;
        cell.textContent = "No model results are available.";
        row.appendChild(cell);
        body.appendChild(row);
        return;
    }

    results.forEach((result) => {
        const row = document.createElement("tr");

        const values = [
            result.model_name,
            formatPrice(result.predicted_price),
            formatMetric(result.mae),
            formatMetric(result.rmse),
            formatMetric(result.baseline_mae),
            result.directional_accuracy == null
                ? "—"
                : `${formatMetric(result.directional_accuracy)}%`
        ];

        values.forEach((value) => {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        });

        body.appendChild(row);
    });
}

async function loadModelComparison() {
    currentCompany = $("companySelect").value;

    setStatus("comparisonStatus", "Comparing all models...", "loading");

    if ($("compareButton")) {
        $("compareButton").disabled = true;
    }

    try {
        const data = await fetchJSON(
            `/api/compare/${encodeURIComponent(currentCompany)}`
        );

        const results = data.results || [];

        renderComparisonTable(results);

        destroyChart(comparisonChart);

        comparisonChart = makeBarChart(
            "comparisonChart",
            results.map((result) => result.model_name),
            [
                {
                    label: "Model MAE",
                    data: results.map((result) => result.mae)
                },
                {
                    label: "Baseline MAE",
                    data: results.map((result) => result.baseline_mae)
                }
            ],
            "Mean absolute error"
        );

        setStatus(
            "comparisonStatus",
            `Compared ${results.length} models for ${currentCompany}.`,
            "success"
        );
    } catch (error) {
        setStatus("comparisonStatus", error.message, "error");
    } finally {
        if ($("compareButton")) {
            $("compareButton").disabled = false;
        }
    }
}

async function loadRiskData() {
    currentCompany = $("companySelect").value;

    setStatus("riskStatus", "Calculating historical risk metrics...", "loading");

    try {
        const data = await fetchJSON(
            `/api/risk/${encodeURIComponent(currentCompany)}`
        );

        setText("cumulativeReturn", formatPercent(data.cumulative_return_pct));
        setText(
            "annualizedVolatility",
            `${formatMetric(data.annualized_volatility_pct)}%`
        );
        setText(
            "maximumDrawdown",
            formatPercent(data.maximum_drawdown_pct)
        );
        setText("sharpeRatio", formatMetric(data.sharpe_ratio, 3));

        setMetricColor("cumulativeReturn", data.cumulative_return_pct);
        setMetricColor("maximumDrawdown", data.maximum_drawdown_pct);

        const dates = data.dates || [];

        destroyChart(cumulativeReturnChart);
        cumulativeReturnChart = makeLineChart(
            "cumulativeReturnChart",
            dates,
            [{
                label: "Cumulative return",
                data: data.cumulative_returns || [],
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.05
            }],
            "Return (%)"
        );

        destroyChart(drawdownChart);
        drawdownChart = makeLineChart(
            "drawdownChart",
            dates,
            [{
                label: "Drawdown",
                data: data.drawdowns || [],
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.05
            }],
            "Drawdown (%)"
        );

        setText("riskNote", data.note || "Historical risk metrics do not guarantee future results.");

        setStatus("riskStatus", `Risk analysis updated for ${currentCompany}.`, "success");
    } catch (error) {
        setStatus("riskStatus", error.message, "error");
    }
}

async function refreshDashboard() {
    await loadCompanyData();
}

function setupEventListeners() {
    $("companySelect")?.addEventListener("change", refreshDashboard);
    $("refreshButton")?.addEventListener("click", refreshDashboard);
    $("predictButton")?.addEventListener("click", loadPrediction);
    $("compareButton")?.addEventListener("click", loadModelComparison);
    $("riskButton")?.addEventListener("click", loadRiskData);

    $("modelSelect")?.addEventListener("change", () => {
        currentModel = $("modelSelect").value;
        loadPrediction();
    });

    document.querySelectorAll(".nav-link").forEach((link) => {
        link.addEventListener("click", () => {
            document.querySelectorAll(".nav-link").forEach((item) => {
                item.classList.remove("active");
            });
            link.classList.add("active");
        });
    });
}

document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
    refreshDashboard();
});