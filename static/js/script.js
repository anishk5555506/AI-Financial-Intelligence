
"use strict";

const API = "/api";

let stockChart = null;
let predictionChart = null;
let cumulativeReturnChart = null;
let drawdownChart = null;

let currentCompany = "Apple";
let requestVersion = 0;

const $ = (id) => document.getElementById(id);

const numberFormat = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2
});

const volumeFormat = new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2
});

function formatNumber(value, digits = 2) {
    const number = Number(value);
    if (value === null || value === undefined || !Number.isFinite(number)) {
        return "—";
    }

    return number.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
    });
}

function formatPrice(value) {
    const number = Number(value);
    if (value === null || value === undefined || !Number.isFinite(number)) {
        return "—";
    }

    return "$" + number.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function formatPercent(value) {
    const number = Number(value);
    if (value === null || value === undefined || !Number.isFinite(number)) {
        return "—";
    }

    return `${number >= 0 ? "+" : ""}${formatNumber(number)}%`;
}

function setText(id, value) {
    const element = $(id);
    if (element) element.textContent = value;
}

function setStatus(id, message, type = "") {
    const element = $(id);
    if (!element) return;

    element.textContent = message;
    element.className = "status-message";

    if (type) element.classList.add(type);
}

function setMetricColor(id, value) {
    const element = $(id);
    if (!element) return;

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

async function fetchJSON(url) {
    const response = await fetch(url, {
        headers: { Accept: "application/json" }
    });

    let result;

    try {
        result = await response.json();
    } catch {
        throw new Error(`Invalid server response (${response.status}).`);
    }

    if (!response.ok) {
        throw new Error(result.error || `Request failed (${response.status}).`);
    }

    return result;
}

function destroyChart(chart) {
    if (chart) chart.destroy();
}

function makeLineChart(canvasId, labels, datasets, options = {}) {
    const canvas = $(canvasId);
    if (!canvas) return null;

    return new Chart(canvas, {
        type: "line",
        data: {
            labels,
            datasets: datasets.map((dataset) => ({
                label: dataset.label,
                data: dataset.data,
                borderColor: dataset.color,
                backgroundColor: dataset.color.replace("1)", "0.10)"),
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0.2,
                fill: Boolean(dataset.fill),
                spanGaps: false
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: "index" },
            plugins: {
                legend: { display: true, position: "top" }
            },
            scales: {
                x: {
                    ticks: { maxTicksLimit: 9, maxRotation: 0 },
                    grid: { display: false }
                },
                y: {
                    grid: { color: "rgba(120, 130, 150, 0.12)" },
                    title: {
                        display: Boolean(options.yTitle),
                        text: options.yTitle || ""
                    },
                    ticks: {
                        callback: options.yTickCallback || ((value) => value)
                    }
                }
            }
        }
    });
}

function renderStockChart(data) {
    destroyChart(stockChart);

    stockChart = makeLineChart("stockChart", data.dates, [{
        label: `${data.company} Adjusted Close`,
        data: data.prices,
        color: "rgba(72, 119, 245, 1)",
        fill: true
    }], {
        yTitle: "Price (USD)",
        yTickCallback: (value) => "$" + numberFormat.format(value)
    });
}

function renderPredictionChart(evaluation) {
    destroyChart(predictionChart);

    predictionChart = makeLineChart(
        "predictionChart",
        evaluation?.dates || [],
        [
            {
                label: "Actual Price",
                data: evaluation?.actual || [],
                color: "rgba(72, 119, 245, 1)"
            },
            {
                label: "Model Prediction",
                data: evaluation?.predicted || [],
                color: "rgba(20, 160, 125, 1)"
            },
            {
                label: "Baseline Prediction",
                data: evaluation?.baseline || [],
                color: "rgba(220, 150, 50, 1)"
            }
        ],
        {
            yTitle: "Price (USD)",
            yTickCallback: (value) => "$" + numberFormat.format(value)
        }
    );
}

function renderRiskCharts(data) {
    destroyChart(cumulativeReturnChart);
    destroyChart(drawdownChart);

    cumulativeReturnChart = makeLineChart(
        "cumulativeReturnChart",
        data.dates || [],
        [{
            label: "Cumulative Return",
            data: data.cumulative_returns || [],
            color: "rgba(20, 160, 125, 1)",
            fill: true
        }],
        {
            yTitle: "Return (%)",
            yTickCallback: (value) => `${numberFormat.format(value)}%`
        }
    );

    drawdownChart = makeLineChart(
        "drawdownChart",
        data.dates || [],
        [{
            label: "Drawdown",
            data: data.drawdowns || [],
            color: "rgba(220, 75, 85, 1)",
            fill: true
        }],
        {
            yTitle: "Drawdown (%)",
            yTickCallback: (value) => `${numberFormat.format(value)}%`
        }
    );
}

function displayRiskMetrics(data) {
    setText("riskPeriod", `${data.start_date} to ${data.end_date}`);
    setText("riskObservations", numberFormat.format(data.observations));
    setText("riskFreeRate", `${formatNumber(data.risk_free_rate_pct)}%`);

    setText("cumulativeReturn", formatPercent(data.cumulative_return_pct));
    setText("annualizedVolatility", `${formatNumber(data.annualized_volatility_pct)}%`);
    setText("maximumDrawdown", `${formatNumber(data.maximum_drawdown_pct)}%`);
    setText("sharpeRatio", formatNumber(data.sharpe_ratio, 3));

    setMetricColor("cumulativeReturn", data.cumulative_return_pct);
    setMetricColor("annualizedVolatility", data.annualized_volatility_pct);
    setMetricColor("maximumDrawdown", data.maximum_drawdown_pct);
    setMetricColor("sharpeRatio", data.sharpe_ratio);

    setText("riskDisclaimerNote", data.note || "Historical estimates do not guarantee future performance.");
    renderRiskCharts(data);

    setStatus("riskStatus", `Risk analysis loaded for ${data.company} (${data.ticker}).`, "success");
}

function displayWalkForward(walkForward) {
    setText("walkForwardMae", formatPrice(walkForward?.mae));
    setText("walkForwardRmse", formatPrice(walkForward?.rmse));
    setText(
        "walkForwardDirection",
        walkForward?.directional_accuracy == null
            ? "Unavailable"
            : `${formatNumber(walkForward.directional_accuracy)}%`
    );

    const folds = walkForward?.folds ?? 0;
    const samples = walkForward?.samples ?? 0;

    setText(
        "walkForwardSamples",
        folds > 0 ? `${folds} folds / ${samples} samples` : "Unavailable"
    );
}

async function loadSelectedModel(expectedVersion = requestVersion) {
    const company = currentCompany;
    const modelId = $("modelSelect").value;

    setStatus("predictionStatus", "Training model and evaluating test data...");

    try {
        const data = await fetchJSON(
            `${API}/predict/${encodeURIComponent(company)}?model=${encodeURIComponent(modelId)}`
        );

        if (expectedVersion !== requestVersion || company !== currentCompany) return;

        setText("predictionCurrentPrice", formatPrice(data.current_price));
        setText("predictedPrice", formatPrice(data.predicted_price));
        setText("mae", formatPrice(data.mae));
        setText("rmse", formatPrice(data.rmse));
        setText("baselineMae", formatPrice(data.baseline_mae));
        setText("baselineRmse", formatPrice(data.baseline_rmse));
        setText("testSamples", numberFormat.format(data.test_samples));
        setText(
            "directionalAccuracy",
            `${formatNumber(data.directional_accuracy)}%`
        );
        setText(
            "baselineDirectionalAccuracy",
            `${formatNumber(data.baseline_directional_accuracy)}%`
        );

        displayWalkForward(data.walk_forward);

        setText("selectedModelName", data.model_name);

        const difference = Number(data.predicted_price) - Number(data.current_price);
        const differencePercent = Number(data.current_price) !== 0
            ? difference / Number(data.current_price) * 100
            : 0;

        const direction = difference > 0 ? "above" : difference < 0 ? "below" : "equal to";

        setText(
            "modelResult",
            `Estimated next close: ${formatPrice(data.predicted_price)}, ` +
            `${formatPercent(differencePercent)} ${direction} the latest close. ` +
            "This is an experimental estimate, not financial advice."
        );

        const baselineBetter =
            Number(data.baseline_mae) < Number(data.mae) &&
            Number(data.baseline_rmse) < Number(data.rmse);

        setText(
            "evaluationNote",
            baselineBetter
                ? "The baseline has lower MAE and RMSE than this model on the holdout test set."
                : "Compare both holdout errors and walk-forward results before judging model usefulness."
        );

        renderPredictionChart(data.evaluation);

        setStatus(
            "predictionStatus",
            `${data.model_name} evaluated on ${data.test_samples} holdout samples.`,
            "success"
        );
    } catch (error) {
        if (expectedVersion !== requestVersion) return;
        setStatus("predictionStatus", error.message, "error");
    }
}

async function compareAllModels() {
    const company = currentCompany;
    const version = requestVersion;
    const button = $("compareButton");

    button.disabled = true;
    button.textContent = "Comparing...";
    setStatus("comparisonStatus", "Training and comparing five models...");

    try {
        const data = await fetchJSON(
            `${API}/compare/${encodeURIComponent(company)}`
        );

        if (version !== requestVersion || company !== currentCompany) return;

        const tbody = $("modelComparisonBody");
        tbody.replaceChildren();

        if (!Array.isArray(data.results) || data.results.length === 0) {
            setStatus("comparisonStatus", "No comparison results returned.", "error");
            return;
        }

        for (const result of data.results) {
            const row = document.createElement("tr");

            const walkForward = result.walk_forward || {};

            const beatsBaseline =
                Number(result.mae) < Number(result.baseline_mae) &&
                Number(result.rmse) < Number(result.baseline_rmse);

            const values = [
                result.model_name,
                formatPrice(result.predicted_price),
                formatPrice(result.mae),
                formatPrice(result.rmse),
                formatPrice(result.baseline_mae),
                formatPrice(result.baseline_rmse),
                result.directional_accuracy == null
                    ? "—"
                    : `${formatNumber(result.directional_accuracy)}%`,
                formatPrice(walkForward.mae),
                formatPrice(walkForward.rmse),
                walkForward.directional_accuracy == null
                    ? "—"
                    : `${formatNumber(walkForward.directional_accuracy)}%`,
                beatsBaseline ? "Yes" : "No"
            ];

            values.forEach((value, index) => {
                const cell = document.createElement("td");
                cell.textContent = value;

                if (index === 10) {
                    cell.className = beatsBaseline ? "positive" : "negative";
                }

                row.appendChild(cell);
            });

            tbody.appendChild(row);
        }

        setStatus(
            "comparisonStatus",
            `Compared ${data.results.length} models for ${company}.`,
            "success"
        );
    } catch (error) {
        if (version !== requestVersion) return;
        setStatus("comparisonStatus", error.message, "error");
    } finally {
        button.disabled = false;
        button.textContent = "Compare All Models";
    }
}

async function loadCompanyData() {
    const company = $("companySelect").value;
    currentCompany = company;
    const version = ++requestVersion;

    setStatus("companyStatus", `Loading ${company} market data...`);
    setStatus("riskStatus", "Calculating risk metrics...");
    setStatus("predictionStatus", "Loading prediction...");
    setStatus("comparisonStatus", "");

    $("predictButton").disabled = true;
    $("compareButton").disabled = true;
    $("riskRefreshButton").disabled = true;

    try {
        const [stock, risk] = await Promise.all([
            fetchJSON(`${API}/company/${encodeURIComponent(company)}`),
            fetchJSON(`${API}/risk/${encodeURIComponent(company)}`)
        ]);

        if (version !== requestVersion) return;

        setText("currentPrice", formatPrice(stock.latest_price));
        setText("priceChange", formatPrice(stock.price_change));
        setText("percentageChange", formatPercent(stock.percentage_change));
        setText("volume", volumeFormat.format(stock.volume));
        setText("trend", stock.trend);

        setMetricColor("priceChange", stock.price_change);
        setMetricColor("percentageChange", stock.percentage_change);

        renderStockChart(stock);

        setText("insightTrend", `${company}: ${stock.trend} Daily Movement`);

        const direction = stock.percentage_change > 0
            ? "increased"
            : stock.percentage_change < 0
                ? "decreased"
                : "remained unchanged";

        setText(
            "insightText",
            `The latest adjusted close is ${formatPrice(stock.latest_price)}. ` +
            `It ${direction} by ${formatPercent(stock.percentage_change)} ` +
            "compared with the previous available session."
        );

        setStatus("companyStatus", `${company} data loaded successfully.`, "success");
        displayRiskMetrics(risk);

        await loadSelectedModel(version);
    } catch (error) {
        if (version !== requestVersion) return;

        setStatus("companyStatus", error.message, "error");
        setStatus("riskStatus", error.message, "error");
    } finally {
        if (version === requestVersion) {
            $("predictButton").disabled = false;
            $("compareButton").disabled = false;
            $("riskRefreshButton").disabled = false;
        }
    }
}

async function refreshRiskAnalytics() {
    const company = currentCompany;
    const version = requestVersion;

    $("riskRefreshButton").disabled = true;
    setStatus("riskStatus", "Refreshing risk calculations...");

    try {
        const data = await fetchJSON(
            `${API}/risk/${encodeURIComponent(company)}`
        );

        if (version !== requestVersion || company !== currentCompany) return;
        displayRiskMetrics(data);
    } catch (error) {
        if (version === requestVersion) {
            setStatus("riskStatus", error.message, "error");
        }
    } finally {
        $("riskRefreshButton").disabled = false;
    }
}

function clearComparison() {
    $("modelComparisonBody").innerHTML =
        '<tr><td colspan="11" class="empty-cell">Select Compare All Models to run the comparison.</td></tr>';
}

document.addEventListener("DOMContentLoaded", () => {
    $("companySelect").addEventListener("change", () => {
        clearComparison();
        loadCompanyData();
    });

    $("modelSelect").addEventListener("change", () => {
        loadSelectedModel(requestVersion);
    });

    $("predictButton").addEventListener("click", () => {
        loadSelectedModel(requestVersion);
    });

    $("compareButton").addEventListener("click", compareAllModels);
    $("riskRefreshButton").addEventListener("click", refreshRiskAnalytics);

    loadCompanyData();
});
