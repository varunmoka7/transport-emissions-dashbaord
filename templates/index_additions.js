function createTimeSeriesChart(timeData, periodType) {
    // Prepare data
    const period = periodType === 'monthly' ? 'Month' : 'Week';
    const periodLabel = periodType === 'monthly' ? 'Month' : 'Week';
    
    const trace1 = {
        x: timeData.map(d => d[period]),
        y: timeData.map(d => d.Total_Emissions_kg),
        name: 'Total Emissions',
        type: 'bar',
        marker: {
            color: '#0d6efd'
        },
        hovertemplate: '<b>%{x}</b><br>Emissions: %{y:.2f} kg CO₂<extra></extra>'
    };
    
    const trace2 = {
        x: timeData.map(d => d[period]),
        y: timeData.map(d => d.Emissions_per_km),
        name: 'Emissions per km',
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: 'y2',
        line: {
            color: '#dc3545',
            width: 3
        },
        marker: {
            size: 8
        },
        hovertemplate: '<b>%{x}</b><br>Emissions/km: %{y:.2f} kg CO₂/km<extra></extra>'
    };
    
    const trace3 = {
        x: timeData.map(d => d[period]),
        y: timeData.map(d => d.Trip_Count),
        name: 'Trip Count',
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: 'y3',
        line: {
            color: '#198754',
            width: 2,
            dash: 'dot'
        },
        marker: {
            size: 6
        },
        hovertemplate: '<b>%{x}</b><br>Trips: %{y}<extra></extra>'
    };
    
    const layout = {
        title: `Emissions Trend by ${periodLabel}`,
        xaxis: {
            title: periodLabel
        },
        yaxis: {
            title: 'Total Emissions (kg CO₂)',
            titlefont: {color: '#0d6efd'},
            tickfont: {color: '#0d6efd'}
        },
        yaxis2: {
            title: 'Emissions per km',
            titlefont: {color: '#dc3545'},
            tickfont: {color: '#dc3545'},
            overlaying: 'y',
            side: 'right',
            showgrid: false
        },
        yaxis3: {
            title: 'Trip Count',
            titlefont: {color: '#198754'},
            tickfont: {color: '#198754'},
            overlaying: 'y',
            side: 'right',
            position: 0.95,
            showgrid: false
        },
        legend: {
            orientation: 'h',
            y: -0.2
        },
        hovermode: 'closest',
        height: 500,
        margin: {t: 50, b: 80}
    };
    
    Plotly.newPlot('timeSeriesChart', [trace1, trace2, trace3], layout);
}

function createEmissionIntensityChart(vehicleData) {
    // Calculate emission intensity (kg CO2 per ton-km)
    vehicleData.forEach(v => {
        if (!v.Avg_Emission_Intensity) {
            v.Avg_Emission_Intensity = v.Total_Emissions_kg / (v.Total_Distance_km * v.Avg_Load_Tons);
        }
    });
    
    // Sort by intensity (lower is better) and take top 15
    vehicleData.sort((a, b) => a.Avg_Emission_Intensity - b.Avg_Emission_Intensity);
    const topVehicles = vehicleData.slice(0, 15);
    
    // Color scale based on efficiency
    const colorScale = topVehicles.map(v => {
        if (v.Avg_Emission_Intensity < 0.05) return '#198754'; // Very good - dark green
        if (v.Avg_Emission_Intensity < 0.08) return '#28a745'; // Good - green
        if (v.Avg_Emission_Intensity < 0.12) return '#ffc107'; // Average - yellow
        if (v.Avg_Emission_Intensity < 0.16) return '#fd7e14'; // Below average - orange
        return '#dc3545'; // Poor - red
    });
    
    const trace = {
        x: topVehicles.map(v => v.Vehicle),
        y: topVehicles.map(v => v.Avg_Emission_Intensity),
        type: 'bar',
        marker: {
            color: colorScale
        },
        hovertemplate: '<b>%{x}</b><br>Emission Intensity: %{y:.4f} kg CO₂/ton-km<br>Trips: %{customdata}<extra></extra>',
        customdata: topVehicles.map(v => v.Trip_Count)
    };
    
    const layout = {
        margin: {t: 10, b: 80},
        yaxis: {
            title: 'Emission Intensity (kg CO₂/ton-km)'
        },
        xaxis: {
            tickangle: -45
        },
        annotations: [{
            text: 'Lower values indicate more efficient transport',
            x: 0.5,
            y: 1.05,
            xref: 'paper',
            yref: 'paper',
            showarrow: false,
            font: {
                size: 12,
                color: '#6c757d'
            }
        }]
    };
    
    Plotly.newPlot('emissionIntensityChart', [trace], layout);
}
