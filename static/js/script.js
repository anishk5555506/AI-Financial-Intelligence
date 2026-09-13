let stockChart;


// Load companies when the page starts
async function loadCompanies() {

    const response = await fetch("/api/companies");

    const companies = await response.json();

    const select = document.getElementById("companySelect");

    select.innerHTML = "";

    companies.forEach(function(company) {

        const option = document.createElement("option");

        option.value = company;
        option.textContent = company;

        select.appendChild(option);

    });

    // Load the first company
    loadCompanyData(companies[0]);
}


// Load selected company's financial data
async function loadCompanyData(company) {

    const response = await fetch("/api/company/" + company);

    const data = await response.json();

    document.getElementById("currentPrice").textContent =
        "$" + data.latest_price;

    document.getElementById("priceChange").textContent =
        data.percentage_change + "%";

    document.getElementById("volume").textContent =
        data.volume.toLocaleString();

    document.getElementById("trend").textContent =
        data.trend;

    document.getElementById("insightTrend").textContent =
        data.trend;


    if (data.trend === "Positive") {

        document.getElementById("insightText").textContent =
            "The stock has increased compared with the previous trading day.";

    }
    else if (data.trend === "Negative") {

        document.getElementById("insightText").textContent =
            "The stock has decreased compared with the previous trading day.";

    }
    else {

        document.getElementById("insightText").textContent =
            "The stock price has remained relatively unchanged.";

    }


    createChart(data);
}


// Create stock price chart
function createChart(data) {

    const ctx = document.getElementById("stockChart");

    if (stockChart) {
        stockChart.destroy();
    }

    stockChart = new Chart(ctx, {

        type: "line",

        data: {

            labels: data.dates,

            datasets: [
                {
                    label: data.company + " Closing Price",

                    data: data.prices,

                    tension: 0.3
                }
            ]

        },

        options: {

            responsive: true,

            scales: {

                y: {
                    beginAtZero: false
                }

            }

        }

    });
}


// Detect company selection
document.getElementById("companySelect")
    .addEventListener("change", function() {

        loadCompanyData(this.value);

    });


// Start application
loadCompanies();