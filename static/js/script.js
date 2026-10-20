"use strict";

let priceChart = null;
let predictionChart = null;
let comparisonChart = null;
let cumulativeReturnChart = null;
let drawdownChart = null;
let rangeChart = null;
let volChart = null;

let currentCompany = "Apple";
let currentModel = "random_forest";

// Results kept in memory so dropdowns can re-render without new requests
let comparisonResults = [];
let rangeData = null;
let volData = null;
let volCompareResults = [];

const $ = (id) => document.getElementById(id);

const numberFormat = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2
});

/* ------------------------------------------------------------------ */
/* Formatting helpers                                                  */
/* ------------------------------------------------------------------ */

function setText(id, value) {
    const element = $(id);
    if (element) {
        element.textContent = value;
    }
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

function formatPercent(value, digits = 2) {
    const number = Number(value);

    if (value === null || value === undefined || !Number.isFinite(number)) {
        return "—";
    }

    return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}%`;
}

function formatMetric(value, digits = 2) {
    if (
        value === null ||
        value === undefined ||
        value === "" ||
        !Number.isFinite(Number(value))
    ) {
        return "—";
    }

    return Number(value).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
    });
}

function formatPlainPercent(value, digits = 2) {
    const text = formatMetric(value, digits);
    return text === "—" ? text : `${text}%`;
}

function formatPValue(value) {
    const number = Number(value);

    if (value === null || value === undefined || !Number.isFinite(number)) {
        return "—";
    }

    return number < 0.001 ? "<0.001" : number.toFixed(3);
}

function horizonText(key) {
    const days = parseInt(String(key), 10);
    return Number.isFinite(days)
        ? `${days} trading day${days === 1 ? "" : "s"}`
        : String(key);
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

function setButtonDisabled(id, disabled) {
    const element = $(id);
    if (element) {
        element.disabled = disabled;
    }
}

async function fetchJSON(url) {
    const response = await fetch(url, {
        headers: {
            Accept: "application/json"
        }
    });

    let payload;

    try {
        payload = await response.json();
    } catch {
        throw new Error(
            `The server returned an invalid response (${response.status}).`
        );
    }

    if (!response.ok) {
        throw new Error(
            payload.error || `Request failed (${response.status}).`
        );
    }

    return payload;
}

/* ------------------------------------------------------------------ */
/* Charts                                                              */
/* ------------------------------------------------------------------ */

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

    const options = commonChartOptions(yTitle);
    options.scales.y.beginAtZero = true;

    return new Chart(canvas, {
        type: "bar",
        data: {
            labels,
            datasets
        },
        options
    });
}

/* ------------------------------------------------------------------ */
/* Dashboard                                                           */
/* ------------------------------------------------------------------ */

async function loadCompanyData() {
    const companySelect = $("companySelect");

    if (!companySelect) {
        return;
    }

    currentCompany = companySelect.value;

    setStatus(
        "globalStatus",
        `Loading ${currentCompany} market data...`,
        "loading"
    );

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
            data.trend === "Positive"
                ? 1
                : data.trend === "Negative"
                    ? -1
                    : 0
        );

        const dates = data.dates || [];

        setText(
            "lastUpdated",
            `Latest available date: ${dates[dates.length - 1] || "—"}`
        );

        destroyChart(priceChart);

        priceChart = makeLineChart(
            "priceChart",
            dates,
            [
                {
                    label: `${currentCompany} closing price`,
                    data: data.prices || [],
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.15
                }
            ],
            "Price (USD)"
        );

        setStatus(
            "globalStatus",
            `${currentCompany} data loaded successfully.`,
            "success"
        );

        await Promise.allSettled([
            loadRiskData(),
            loadPrediction(),
            loadRanges(),
            loadVolatility()
        ]);
    } catch (error) {
        setStatus("globalStatus", error.message, "error");
    }
}

/* ------------------------------------------------------------------ */
/* Price forecasts                                                     */
/* ------------------------------------------------------------------ */

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
            dateId: "forecast5Date",
            fallbackDate: "Five trading days ahead"
        },
        {
            key: "10_day",
            priceId: "forecast10Price",
            returnId: "forecast10Return",
            dateId: "forecast10Date",
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
            setText(
                card.dateId,
                result.horizon_label || card.fallbackDate
            );
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
    const companySelect = $("companySelect");
    const modelSelect = $("modelSelect");

    if (!companySelect || !modelSelect) {
        return;
    }

    currentCompany = companySelect.value;
    currentModel = modelSelect.value;

    setStatus(
        "predictionStatus",
        "Training models and generating forecasts...",
        "loading"
    );

    setButtonDisabled("predictButton", true);

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
        setButtonDisabled("predictButton", false);
    }
}

/* ------------------------------------------------------------------ */
/* Price model comparison (with horizon selector)                      */
/* ------------------------------------------------------------------ */

function renderComparisonTable(rows) {
    const body = $("comparisonTableBody");

    if (!body) {
        console.error(
            'Comparison table body not found. Check for id="comparisonTableBody" in index.html.'
        );

        setStatus(
            "comparisonStatus",
            "Comparison table could not be found in index.html.",
            "error"
        );

        return;
    }

    body.replaceChildren();

    const columnCount = 13;

    if (!Array.isArray(rows) || rows.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = columnCount;
        cell.textContent = "No model results are available.";

        row.appendChild(cell);
        body.appendChild(row);

        return;
    }

    rows.forEach(({ result, h }) => {
        const row = document.createElement("tr");
        const wf = h.walk_forward || {};

        // Verdict uses WALK-FORWARD results (many more test days than the holdout)
        let verdict = "—";

        if (typeof wf.beats_baseline === "boolean") {
            if (!wf.beats_baseline) {
                verdict = "No";
            } else if (wf.dm_p_value != null && wf.dm_p_value < 0.05) {
                verdict = "Yes (significant)";
            } else {
                verdict = "Yes (not significant)";
            }
        }

        const values = [
            result.model_name || result.model_id || "Unknown model",
            formatPrice(h.predicted_price),
            formatMetric(h.mae),
            formatMetric(h.rmse),
            formatMetric(h.baseline_mae),
            formatMetric(h.baseline_rmse),
            formatPlainPercent(h.directional_accuracy),
            formatMetric(wf.mae),
            formatMetric(wf.rmse),
            formatMetric(wf.baseline_mae),
            formatMetric(wf.baseline_rmse),
            formatPlainPercent(wf.directional_accuracy),
            verdict
        ];

        values.forEach((value, index) => {
            const cell = document.createElement("td");

            cell.textContent = value;
            cell.dataset.column = String(index + 1);

            if (index === 0) {
                cell.classList.add("model-name-cell");
            }

            if (index === 12) {
                cell.classList.add(
                    value.startsWith("Yes")
                        ? "positive"
                        : value === "No"
                            ? "negative"
                            : "neutral"
                );
            }

            row.appendChild(cell);
        });

        body.appendChild(row);
    });
}

function renderComparisonView() {
    const key = $("compareHorizonSelect")?.value || "1_day";

    const rows = comparisonResults
        .map((result) => ({ result, h: result.horizons?.[key] }))
        .filter((item) => item.h)
        .sort((a, b) => {
            const aMae = a.h.walk_forward?.mae ?? Infinity;
            const bMae = b.h.walk_forward?.mae ?? Infinity;
            return aMae - bMae;
        });

    renderComparisonTable(rows);

    destroyChart(comparisonChart);

    comparisonChart = makeBarChart(
        "comparisonChart",
        rows.map(({ result }) => result.model_name || result.model_id),
        [
            {
                label: "Model MAE (walk-forward)",
                data: rows.map(({ h }) => h.walk_forward?.mae ?? null)
            },
            {
                label: "Baseline MAE (walk-forward)",
                data: rows.map(({ h }) => h.walk_forward?.baseline_mae ?? null)
            }
        ],
        `Mean absolute error (${horizonText(key)})`
    );
}

async function loadModelComparison() {
    const companySelect = $("companySelect");

    if (!companySelect) {
        return;
    }

    currentCompany = companySelect.value;

    setStatus(
        "comparisonStatus",
        "Comparing all models at every horizon (the first run can take a minute)...",
        "loading"
    );

    setButtonDisabled("compareButton", true);

    try {
        const data = await fetchJSON(
            `/api/compare/${encodeURIComponent(currentCompany)}`
        );

        comparisonResults = Array.isArray(data.results)
            ? data.results
            : [];

        renderComparisonView();

        setStatus(
            "comparisonStatus",
            `Compared ${comparisonResults.length} models for ${currentCompany}. Use the horizon dropdown to switch between 1, 5 and 10 days.`,
            "success"
        );
    } catch (error) {
        setStatus("comparisonStatus", error.message, "error");
    } finally {
        setButtonDisabled("compareButton", false);
    }
}

/* ------------------------------------------------------------------ */
/* Prediction ranges                                                   */
/* ------------------------------------------------------------------ */

function renderRangeCards(horizons) {
    const cards = [
        { key: "1_day", id: "range1" },
        { key: "5_day", id: "range5" },
        { key: "10_day", id: "range10" }
    ];

    cards.forEach(({ key, id }) => {
        const item = horizons && horizons[key];

        if (!item) {
            setText(`${id}Value`, "—");
            setText(`${id}Width`, "—");
            setText(`${id}Method`, "Not available");
            return;
        }

        const method = item.methods?.[item.headline_method] || {};

        setText(
            `${id}Value`,
            `${formatPrice(item.lower_price)} – ${formatPrice(item.upper_price)}`
        );

        setText(
            `${id}Width`,
            `Width: ${formatPlainPercent(item.width_pct, 1)} of the price`
        );

        setText(
            `${id}Method`,
            method.coverage == null
                ? item.headline_name
                : `${item.headline_name} · past coverage ${formatPlainPercent(method.coverage, 1)}`
        );
    });
}

function renderRangeTable(horizons) {
    const body = $("rangeTableBody");

    if (!body) {
        return;
    }

    body.replaceChildren();

    const rows = [];

    ["1_day", "5_day", "10_day"].forEach((key) => {
        const item = horizons && horizons[key];

        if (!item) {
            return;
        }

        Object.entries(item.methods || {}).forEach(([methodKey, method]) => {
            rows.push({ item, methodKey, method });
        });
    });

    if (rows.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 6;
        cell.textContent = "No range results are available.";

        row.appendChild(cell);
        body.appendChild(row);

        return;
    }

    rows.forEach(({ item, methodKey, method }) => {
        const row = document.createElement("tr");

        const values = [
            horizonText(item.horizon),
            method.name,
            formatPlainPercent(method.coverage, 1),
            formatPlainPercent(method.width_pct, 1),
            method.verdict || "—",
            item.headline_method === methodKey ? "Yes" : ""
        ];

        values.forEach((value, index) => {
            const cell = document.createElement("td");

            cell.textContent = value;

            if (index === 0) {
                cell.classList.add("model-name-cell");
            }

            if (index === 4) {
                cell.classList.add(
                    value === "Well calibrated"
                        ? "positive"
                        : value === "Too narrow"
                            ? "negative"
                            : "neutral"
                );
            }

            row.appendChild(cell);
        });

        body.appendChild(row);
    });
}

function renderRangeChart() {
    destroyChart(rangeChart);
    rangeChart = null;

    const key = $("rangeHorizonSelect")?.value || "1_day";
    const item = rangeData?.[key];
    const evaluation = item?.evaluation;

    if (!evaluation || !evaluation.dates || evaluation.dates.length === 0) {
        return;
    }

    rangeChart = makeLineChart(
        "rangeChart",
        evaluation.dates,
        [
            {
                label: "Actual price",
                data: evaluation.actual,
                borderColor: "#f8fafc",
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.05
            },
            {
                label: "Lower end of range",
                data: evaluation.lower,
                borderColor: "rgba(56, 189, 248, 0.7)",
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.05,
                fill: false
            },
            {
                label: `Upper end of range (${item.range_level}%)`,
                data: evaluation.upper,
                borderColor: "rgba(56, 189, 248, 0.7)",
                backgroundColor: "rgba(56, 189, 248, 0.18)",
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.05,
                fill: "-1"
            }
        ],
        "Price (USD)"
    );
}

async function loadRanges() {
    const companySelect = $("companySelect");

    if (!companySelect) {
        return;
    }

    currentCompany = companySelect.value;

    setStatus(
        "rangeStatus",
        "Calculating prediction ranges...",
        "loading"
    );

    setButtonDisabled("rangeButton", true);

    try {
        const data = await fetchJSON(
            `/api/range/${encodeURIComponent(currentCompany)}`
        );

        rangeData = data.horizons || {};

        renderRangeCards(rangeData);
        renderRangeTable(rangeData);
        renderRangeChart();

        setStatus(
            "rangeStatus",
            `Prediction ranges updated for ${currentCompany}.`,
            "success"
        );
    } catch (error) {
        setStatus("rangeStatus", error.message, "error");
    } finally {
        setButtonDisabled("rangeButton", false);
    }
}

/* ------------------------------------------------------------------ */
/* Volatility forecasting                                              */
/* ------------------------------------------------------------------ */

function renderVolCards(horizons) {
    const cards = [
        { key: "5_day", id: "vol5" },
        { key: "10_day", id: "vol10" },
        { key: "20_day", id: "vol20" }
    ];

    cards.forEach(({ key, id }) => {
        const item = horizons && horizons[key];

        if (!item) {
            setText(`${id}Value`, "—");
            setText(`${id}Change`, "—");
            setText(`${id}Detail`, "—");
            return;
        }

        setText(`${id}Value`, formatPlainPercent(item.forecast_vol, 1));

        setText(
            `${id}Change`,
            item.change_pct == null
                ? "—"
                : `${formatPercent(item.change_pct, 1)} vs recent`
        );

        // More volatility = more risk, so an increase is shown in red
        setMetricColor(
            `${id}Change`,
            item.change_pct == null ? null : -item.change_pct
        );

        setText(
            `${id}Detail`,
            `Last ${item.horizon} days: ${formatPlainPercent(item.current_vol, 1)}`
        );
    });
}

function volVerdictText(wf) {
    if (!wf || typeof wf.beats_baseline !== "boolean") {
        return { text: "Not enough data to evaluate this model.", type: "" };
    }

    if (!wf.beats_baseline) {
        return {
            text: "This model does not beat the baseline at this horizon.",
            type: "error"
        };
    }

    if (wf.dm_p_value != null && wf.dm_p_value < 0.05) {
        return {
            text: "This model beats the baseline, and the improvement is statistically significant (unlikely to be luck).",
            type: "success"
        };
    }

    return {
        text: "This model is slightly better than the baseline, but the improvement is not statistically significant.",
        type: ""
    };
}

function renderVolDetail() {
    const key = $("volHorizonSelect")?.value || "5_day";
    const item = volData?.[key];

    destroyChart(volChart);
    volChart = null;

    if (!item) {
        return;
    }

    const wf = item.walk_forward || {};

    setText("volModelMAE", formatMetric(wf.mae));
    setText("volBaselineMAE", formatMetric(wf.baseline_mae));
    setText("volImprovement", formatPercent(wf.mae_improvement_pct, 1));
    setText("volPValue", formatPValue(wf.dm_p_value));

    setMetricColor("volImprovement", wf.mae_improvement_pct);

    const verdict = volVerdictText(wf);
    setStatus("volVerdict", verdict.text, verdict.type);

    const evaluation = item.evaluation;

    if (!evaluation || !evaluation.dates || evaluation.dates.length === 0) {
        return;
    }

    volChart = makeLineChart(
        "volChart",
        evaluation.dates,
        [
            {
                label: "Actual volatility",
                data: evaluation.actual,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            },
            {
                label: "Model forecast",
                data: evaluation.predicted,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            },
            {
                label: "Baseline (last period)",
                data: evaluation.baseline,
                borderWidth: 1.5,
                pointRadius: 0,
                borderDash: [5, 5],
                tension: 0.1
            }
        ],
        "Annualized volatility (%)"
    );
}

async function loadVolatility() {
    const companySelect = $("companySelect");
    const modelSelect = $("volModelSelect");

    if (!companySelect || !modelSelect) {
        return;
    }

    currentCompany = companySelect.value;

    setStatus(
        "volStatus",
        "Training volatility models...",
        "loading"
    );

    setButtonDisabled("volButton", true);

    try {
        const query = new URLSearchParams({ model: modelSelect.value });

        const data = await fetchJSON(
            `/api/volatility/${encodeURIComponent(currentCompany)}?${query.toString()}`
        );

        volData = data.horizons || {};

        renderVolCards(volData);
        renderVolDetail();

        setStatus(
            "volStatus",
            `${data.model_name} volatility forecasts generated for ${currentCompany}.`,
            "success"
        );
    } catch (error) {
        setStatus("volStatus", error.message, "error");
    } finally {
        setButtonDisabled("volButton", false);
    }
}

function renderVolComparisonTable() {
    const body = $("volCompareTableBody");

    if (!body) {
        return;
    }

    body.replaceChildren();

    const key = $("volHorizonSelect")?.value || "5_day";

    const rows = volCompareResults
        .map((result) => ({ result, h: result.horizons?.[key] }))
        .filter((item) => item.h)
        .sort((a, b) => {
            const aGain = a.h.walk_forward?.mae_improvement_pct ?? -Infinity;
            const bGain = b.h.walk_forward?.mae_improvement_pct ?? -Infinity;
            return bGain - aGain;
        });

    if (rows.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 8;
        cell.textContent = "No volatility results are available.";

        row.appendChild(cell);
        body.appendChild(row);

        return;
    }

    rows.forEach(({ result, h }) => {
        const row = document.createElement("tr");
        const wf = h.walk_forward || {};

        let verdict = "—";

        if (typeof wf.beats_baseline === "boolean") {
            if (!wf.beats_baseline) {
                verdict = "No";
            } else if (wf.dm_p_value != null && wf.dm_p_value < 0.05) {
                verdict = "Yes (significant)";
            } else {
                verdict = "Yes (not significant)";
            }
        }

        const values = [
            result.model_name || result.model_id,
            formatPlainPercent(h.forecast_vol, 1),
            formatPlainPercent(h.current_vol, 1),
            formatMetric(wf.mae),
            formatMetric(wf.baseline_mae),
            formatPercent(wf.mae_improvement_pct, 1),
            formatPValue(wf.dm_p_value),
            verdict
        ];

        values.forEach((value, index) => {
            const cell = document.createElement("td");

            cell.textContent = value;

            if (index === 0) {
                cell.classList.add("model-name-cell");
            }

            if (index === 7) {
                cell.classList.add(
                    value.startsWith("Yes")
                        ? "positive"
                        : value === "No"
                            ? "negative"
                            : "neutral"
                );
            }

            row.appendChild(cell);
        });

        body.appendChild(row);
    });
}

async function loadVolatilityComparison() {
    const companySelect = $("companySelect");

    if (!companySelect) {
        return;
    }

    currentCompany = companySelect.value;

    setStatus(
        "volCompareStatus",
        "Comparing all volatility models (the first run can take a minute)...",
        "loading"
    );

    setButtonDisabled("volCompareButton", true);

    try {
        const data = await fetchJSON(
            `/api/volatility-compare/${encodeURIComponent(currentCompany)}`
        );

        volCompareResults = Array.isArray(data.results)
            ? data.results
            : [];

        renderVolComparisonTable();

        setStatus(
            "volCompareStatus",
            `Compared ${volCompareResults.length} volatility models for ${currentCompany}. Use the horizon dropdown above to switch horizons.`,
            "success"
        );
    } catch (error) {
        setStatus("volCompareStatus", error.message, "error");
    } finally {
        setButtonDisabled("volCompareButton", false);
    }
}

/* ------------------------------------------------------------------ */
/* Risk                                                                */
/* ------------------------------------------------------------------ */

async function loadRiskData() {
    const companySelect = $("companySelect");

    if (!companySelect) {
        return;
    }

    currentCompany = companySelect.value;

    setStatus(
        "riskStatus",
        "Calculating historical risk metrics...",
        "loading"
    );

    try {
        const data = await fetchJSON(
            `/api/risk/${encodeURIComponent(currentCompany)}`
        );

        setText(
            "cumulativeReturn",
            formatPercent(data.cumulative_return_pct)
        );

        setText(
            "annualizedVolatility",
            `${formatMetric(data.annualized_volatility_pct)}%`
        );

        setText(
            "maximumDrawdown",
            formatPercent(data.maximum_drawdown_pct)
        );

        setText(
            "sharpeRatio",
            formatMetric(data.sharpe_ratio, 3)
        );

        setMetricColor(
            "cumulativeReturn",
            data.cumulative_return_pct
        );

        setMetricColor(
            "maximumDrawdown",
            data.maximum_drawdown_pct
        );

        const dates = data.dates || [];

        destroyChart(cumulativeReturnChart);

        cumulativeReturnChart = makeLineChart(
            "cumulativeReturnChart",
            dates,
            [
                {
                    label: "Cumulative return",
                    data: data.cumulative_returns || [],
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.05
                }
            ],
            "Return (%)"
        );

        destroyChart(drawdownChart);

        drawdownChart = makeLineChart(
            "drawdownChart",
            dates,
            [
                {
                    label: "Drawdown",
                    data: data.drawdowns || [],
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.05
                }
            ],
            "Drawdown (%)"
        );

        setText(
            "riskNote",
            data.note ||
                "Historical risk metrics do not guarantee future results."
        );

        setStatus(
            "riskStatus",
            `Risk analysis updated for ${currentCompany}.`,
            "success"
        );
    } catch (error) {
        setStatus("riskStatus", error.message, "error");
    }
}

/* ------------------------------------------------------------------ */
/* Page wiring                                                         */
/* ------------------------------------------------------------------ */

async function refreshDashboard() {
    // Old comparison tables belong to the previous company
    comparisonResults = [];
    volCompareResults = [];
    renderComparisonTable([]);
    renderVolComparisonTable();

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

    // Horizon dropdowns re-render stored results (no new request)
    $("compareHorizonSelect")?.addEventListener("change", () => {
        if (comparisonResults.length > 0) {
            renderComparisonView();
        }
    });

    $("rangeButton")?.addEventListener("click", loadRanges);
    $("rangeHorizonSelect")?.addEventListener("change", renderRangeChart);

    $("volButton")?.addEventListener("click", loadVolatility);
    $("volModelSelect")?.addEventListener("change", loadVolatility);
    $("volHorizonSelect")?.addEventListener("change", () => {
        renderVolDetail();
        renderVolComparisonTable();
    });
    $("volCompareButton")?.addEventListener("click", loadVolatilityComparison);

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