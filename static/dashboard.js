
// Initialize global variables
let emissionsData = [];
let tripData = [];
let filters = {
    dateRange: null,
    consignee: null,
    vehicle: null,
    status: null
};

// Fetch data from the server
async function fetchData(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching data from ${endpoint}:`, error);
    }
}

// Update KPI values
function updateKPIs(data) {
    document.getElementById('totalEmissions').textContent = data.totalEmissions;
    document.getElementById('emissionIntensity').textContent = data.emissionIntensity;
    document.getElementById('totalTrips').textContent = data.totalTrips;
    document.getElementById('onTimeDelivery').textContent = `${data.onTimeDeliveryRate}%`;
    document.getElementById('emissionsPerTrip').textContent = data.emissionsPerTrip;
    document.getElementById('potentialSavings').textContent = data.potentialSavings;
    document.getElementById('avgDistance').textContent = data.avgDistance;
    document.getElementById('deliveryEfficiency').textContent = data.deliveryEfficiency;
}

// Render charts
function renderCharts(data) {
    // Emissions & Trips Over Time
    Plotly.newPlot('emissionsTimeChart', data.emissionsTimeChart, { responsive: true });

    // Delivery Status Breakdown
    Plotly.newPlot('deliveryStatusChart', data.deliveryStatusChart, { responsive: true });

    // Top 10 Vehicles by Emissions
    Plotly.newPlot('topVehiclesChart', data.topVehiclesChart, { responsive: true });

    // Top 10 Routes by Distance
    Plotly.newPlot('topRoutesChart', data.topRoutesChart, { responsive: true });

    // Additional charts can be added here
}

// Apply filters
function applyFilters() {
    filters.dateRange = document.getElementById('dateRange').value;
    filters.consignee = document.getElementById('consigneeFilter').value;
    filters.vehicle = document.getElementById('vehicleFilter').value;
    filters.status = document.getElementById('statusFilter').value;

    // Fetch filtered data and update dashboard
    fetchData(`/api/dashboard-data?${new URLSearchParams(filters)}`)
        .then(data => {
            updateKPIs(data.kpis);
            renderCharts(data.charts);
        });
}

// Reset filters
function resetFilters() {
    document.getElementById('dateRange').value = '';
    document.getElementById('consigneeFilter').value = '';
    document.getElementById('vehicleFilter').value = '';
    document.getElementById('statusFilter').value = '';
    applyFilters();
}

// Initialize dashboard
function initializeDashboard() {
    // Fetch initial data
    fetchData('/api/dashboard-data')
        .then(data => {
            emissionsData = data.emissions;
            tripData = data.trips;
            updateKPIs(data.kpis);
            renderCharts(data.charts);
        });

    // Attach event listeners
    document.getElementById('applyFilters').addEventListener('click', applyFilters);
    document.getElementById('resetFilters').addEventListener('click', resetFilters);
}

// Run initialization on page load
document.addEventListener('DOMContentLoaded', initializeDashboard);
