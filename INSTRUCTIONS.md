# Greenko Transport Carbon Emissions Dashboard

This web application calculates and visualizes carbon emissions data from Greenko transportation activities.

## Installation

1. Ensure Python 3.8+ is installed on your system
2. Install required dependencies:
```
pip install -r requirements.txt
```

## Running the Application

1. Navigate to the project directory:
```
cd /path/to/PROJECTS
```

2. Start the Flask application:
```
python app.py
```

3. Open your web browser and go to:
```
http://127.0.0.1:5002
```

## Features

- Carbon emissions calculation based on distance, vehicle type, and cargo weight
- Analysis of delivery status (on-time vs delayed) and impact on emissions
- Detailed breakdown of emissions by vehicle and consignee
- Trip-level details with emissions factors
- Visual charts and graphs for data interpretation

## Data Sources

The application uses data from:
- PRL-GreenkoReport-24-25.csv

## Methodology

Carbon emissions are calculated using the following factors:
1. Base emissions: Distance × Emission factor (1.1 kg CO₂/km)
2. Weight adjustment: Heavier loads increase emissions proportionally
3. Speed efficiency: Optimal efficiency at 60-80 km/h; lower/higher speeds increase emissions
4. Idling emissions: 2.5 kg CO₂ per hour of idle time
5. Delay factor: Delayed shipments have 20% higher emissions due to inefficient routing/scheduling

## Future Enhancements

- Integration with real-time vehicle tracking data
- Machine learning for predictive emission modeling
- Route optimization for emissions reduction
- Integration with other ESG metrics
