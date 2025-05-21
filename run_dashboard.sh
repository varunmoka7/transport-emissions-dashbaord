#!/bin/bash
# Script to start the Carbon Emissions Dashboard application

echo "Starting Transport Carbon Emissions Dashboard..."
echo "Opening web browser to http://127.0.0.1:5002"

# Open the browser (works on macOS)
open http://127.0.0.1:5002

# Start the Flask application
python3 app.py
