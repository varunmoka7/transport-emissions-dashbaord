from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import re
from werkzeug.utils import secure_filename
from flask_caching import Cache

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}

# Initialize cache
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global variable to track the current data file
current_data_file = 'PRL-GreenkoReport-24-25.csv'  # Default file

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load and preprocess data
@cache.cached(timeout=300, key_prefix='processed_data')
def load_data():
    """Load data from the current_data_file"""
    global current_data_file
    
    # Determine file extension
    file_ext = current_data_file.rsplit('.', 1)[1].lower()
    
    if file_ext == 'csv':
        df = pd.read_csv(current_data_file)
    elif file_ext == 'xlsx':
        df = pd.read_excel(current_data_file)
    else:
        raise ValueError("Unsupported file format: {}".format(file_ext))
        
    # Convert date columns to datetime
    date_columns = [col for col in df.columns if 'Date' in col or 'At' in col]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            pass
    
    # Extract weight from Consignment column
    df['Estimated_Weight_Tons'] = df['Consignment'].apply(estimate_weight)
    
    # Extract month and year for time series analysis
    df['Trip_Date'] = pd.to_datetime(df['LR Date'], errors='coerce')
    df['Trip_Month'] = df['Trip_Date'].dt.strftime('%Y-%m')
    df['Trip_Week'] = df['Trip_Date'].dt.strftime('%Y-%U')
    
    # Calculate emissions
    df = calculate_emissions(df)
    
    # Calculate additional KPIs
    # Emission intensity per ton-km (logistics industry standard metric)
    df['Emission_Intensity'] = df['Total_Emissions_kg'] / (df['Distance_Covered_Clean'] * df['Estimated_Weight_Tons'])
    
    # Calculate delivery efficiency (1.0 = perfect, higher is worse)
    df['Delivery_Efficiency'] = 1.0
    mask_delayed = df['Transit Status'] == 'Delayed'
    if 'Expected Delivery Date' in df.columns and 'Trip Completed At' in df.columns:
        delayed_rows = df.loc[mask_delayed, ['Expected Delivery Date', 'Trip Completed At']].dropna()
        delayed_rows['Days_Late'] = (pd.to_datetime(delayed_rows['Trip Completed At']) - pd.to_datetime(delayed_rows['Expected Delivery Date'])).dt.days
        df.loc[delayed_rows.index, 'Delivery_Efficiency'] = 1.0 + (delayed_rows['Days_Late'] * 0.2)
                    
    return df

def estimate_weight(consignment_desc):
    """Estimate weight based on consignment description using detailed renewable energy component data"""
    if pd.isna(consignment_desc):
        return 10.0  # Default weight if no description
        
    desc = str(consignment_desc).lower()
    
    # Wind turbine components (more detailed breakdown)
    if 'wind electricity generator' in desc:
        # Complete wind turbine generator set
        return 70.0  # ~70 tons for a complete wind generator
    elif 'generator' in desc and 'wind' in desc:
        # Generator component only
        return 65.0  # ~65 tons for generator component
    elif 'gear box' in desc or 'gearbox' in desc:
        return 25.0  # Approx 25 tons for a gearbox
    elif 'blade' in desc:
        # Wind turbine blade (depends on size)
        if 'large' in desc or 'main' in desc:
            return 18.0  # Large blade ~18 tons
        else:
            return 12.0  # Standard blade ~12 tons
    elif 'tower' in desc:
        # Tower sections
        if 'base' in desc or 'section' in desc:
            return 45.0  # Base tower section
        elif 'middle' in desc:
            return 35.0  # Middle section
        elif 'top' in desc:
            return 25.0  # Top section
        else:
            return 35.0  # Generic tower component
    elif 'nacelle' in desc:
        return 55.0  # Nacelle housing ~55 tons
    elif 'hub' in desc:
        return 20.0  # Hub component ~20 tons
    elif 'transformer' in desc:
        # Transformer components
        if 'dry type' in desc:
            return 15.0  # Dry-type transformer
        elif 'large' in desc or 'power' in desc:
            return 25.0  # Large power transformer
        else:
            return 12.0  # Standard transformer
    elif 'sky lift' in desc:
        # Maintenance equipment
        return 20.0  # Sky lift ~20 tons
    elif 'converter' in desc or 'control' in desc:
        # Control modules
        if 'module' in desc:
            return 5.0  # Converter control module
        else:
            return 8.0  # Converter system
    elif 'igbt' in desc:
        # Power electronics
        if 'stack' in desc or 'array' in desc:
            return 3.0  # IGBT stack
        else:
            return 1.5  # Individual IGBT components
    elif 'failed' in desc:
        # Failed components (usually returning for repair)
        # Weight may be less than new components due to missing parts
        if 'generator' in desc:
            return 60.0  # Failed generator
        elif 'blade' in desc:
            return 10.0  # Failed blade
        else:
            return 15.0  # Generic failed component
    else:
        # Default weights based on quantity information
        quantity_match = re.search(r'(\d+)\s*(?:nos|units|pcs)', desc, re.IGNORECASE)
        if quantity_match:
            quantity = int(quantity_match.group(1))
            if quantity > 5:
                # Multiple small components
                return 5.0 * (quantity / 10.0)  # Scale based on quantity
            elif quantity > 1:
                return 8.0
        
        # Default fallback
        return 10.0  # Default weight if no specific category

def calculate_emissions(df):
    """Calculate carbon emissions based on vehicle data"""
    # Load Indian emission factors
    try:
        emission_factors_df = pd.read_csv('india emissions fatcors for vehicles - Sheet1.csv')
        # Extract Freight Vehicle HDV emission factor (standard for heavy-duty trucks)
        hdv_row = emission_factors_df[emission_factors_df['Vehicle_Type'] == 'Freight Vehicle']
        if not hdv_row.empty:
            BASE_EMISSION_FACTOR = hdv_row.iloc[0]['Emission_Factor_Value']
        else:
            BASE_EMISSION_FACTOR = 0.7375  # Default HDV emission factor if not found
        print(f"Using base emission factor: {BASE_EMISSION_FACTOR} kg CO2/km")
    except Exception as e:
        print(f"Error loading emission factors: {e}")
        BASE_EMISSION_FACTOR = 0.7375  # Default if CSV not found
        
    # Calculate idling emission factor based on HDV standards
    IDLING_EMISSION_FACTOR = 3.2  # kg CO2 per hour for HDV
    
    # Vehicle-specific emission factors based on registration region
    # Apply regional adjustments based on fleet age and maintenance standards
    vehicle_factors = {
        # Format: 'vehicle_prefix': emission_factor
        'RJ': BASE_EMISSION_FACTOR * 0.95,  # Rajasthan (5% better than average - newer fleet)
        'MH': BASE_EMISSION_FACTOR * 1.05,  # Maharashtra (5% worse - older vehicles in freight)
        'AP': BASE_EMISSION_FACTOR * 0.98,  # Andhra Pradesh 
        'NL': BASE_EMISSION_FACTOR * 1.02,  # Nagaland
        'GJ': BASE_EMISSION_FACTOR * 0.97,  # Gujarat (3% better - good maintenance)
        'DD': BASE_EMISSION_FACTOR,         # Delhi/Default
        'TN': BASE_EMISSION_FACTOR * 0.96,  # Tamil Nadu
        'KA': BASE_EMISSION_FACTOR * 1.01   # Karnataka
    }
    
    # Calculate basic emissions based on distance and vehicle type
    # More sophisticated vehicle classification
    def get_vehicle_factor(vehicle_no):
        """
        Determine vehicle emission factor based on vehicle number and known classifications
        Uses both state registration and vehicle type information
        """
        try:
            if pd.isna(vehicle_no):
                return BASE_EMISSION_FACTOR
            
            vehicle_str = str(vehicle_no).lower()
            
            # Extract state prefix (first 2 chars typically)
            prefix = vehicle_no[:2]
            base_factor = vehicle_factors.get(prefix, BASE_EMISSION_FACTOR)
            
            # Apply adjustments based on vehicle type indicators in the registration
            # Age-based adjustments
            current_year = 2025  # Assuming current year is 2025
            
            # Try to extract year from common formats in Indian registrations
            year_match = re.search(r'20(\d{2})', vehicle_str)
            if year_match:
                vehicle_year = 2000 + int(year_match.group(1))
                age = current_year - vehicle_year
                
                # Adjust emission factor based on age
                if age <= 5:  # Newer vehicles, better emission control
                    base_factor *= 0.9
                elif age <= 10:  # Mid-life vehicles
                    base_factor *= 1.0  # No adjustment
                elif age <= 15:  # Older vehicles
                    base_factor *= 1.15
                else:  # Very old vehicles
                    base_factor *= 1.35
            
            # Vehicle type adjustments based on common classifications in registration numbers
            if any(x in vehicle_str for x in ['lcv', 'pick', 'small']):
                # Light commercial vehicles - use LDV emission factor
                try:
                    ldv_factor = emission_factors_df[
                        (emission_factors_df['Vehicle_Type'] == 'Freight Vehicle') & 
                        (emission_factors_df['Category_Type'] == 'LDV')
                    ]['Emission_Factor_Value'].iloc[0]
                    return ldv_factor
                except:
                    return base_factor * 0.7  # Approx 30% lower than HDV
                    
            elif any(x in vehicle_str for x in ['hcv', 'heavy', 'trailer']):
                # Heavy commercial vehicles - use HDV highest factor
                return base_factor * 1.1  # 10% higher than base
            
            # Default is to return the state-based factor
            return base_factor
            
        except Exception as e:
            print(f"Error in vehicle factor calculation: {e}")
            return BASE_EMISSION_FACTOR
    
    # Apply vehicle-specific emission factors
    df['Vehicle_Factor'] = df['Current Vehicle No.'].apply(get_vehicle_factor)
    
    # Calculate basic emissions based on distance covered and vehicle factor
    # Handle missing distance data
    df['Distance_Covered_Clean'] = pd.to_numeric(df['Distance Covered'], errors='coerce').fillna(0)
    
    # Quality check on distances - flag suspiciously short or long distances
    avg_distance = df['Distance_Covered_Clean'].median()
    mask_suspicious = (df['Distance_Covered_Clean'] < 10) | (df['Distance_Covered_Clean'] > 3000)
    
    # For suspicious distances, try to estimate from source-destination if available
    for idx, row in df[mask_suspicious].iterrows():
        if pd.notna(row['Source']) and pd.notna(row['Destination']):
            # This is just a placeholder - in a real-world scenario, you could use 
            # geocoding and routing APIs to estimate the distance
            # For now we'll use the median as a reasonable fallback
            df.loc[idx, 'Distance_Covered_Clean'] = avg_distance
    
    # Calculate base emissions
    df['Base_Emissions_kg'] = df['Distance_Covered_Clean'] * df['Vehicle_Factor']
    
    # Apply weight factor (heavier loads = more emissions)
    df['Weight_Factor'] = 1 + (df['Estimated_Weight_Tons'] / 100)
    
    # Apply efficiency factor based on average speed
    # Extract and clean average speed data
    try:
        df['Avg_Speed_kmh'] = pd.to_numeric(df['Average Speed'].astype(str).str.replace(' km/hr', '').str.replace('km/h', ''), errors='coerce')
        
        # Apply reasonable constraints and handle missing values
        mask_too_low = df['Avg_Speed_kmh'] < 10  # Unrealistically low speed
        mask_too_high = df['Avg_Speed_kmh'] > 120  # Unrealistically high speed
        mask_na = df['Avg_Speed_kmh'].isna()
        
        # Replace unrealistic values with estimated speeds based on distance and time
        distance_col = pd.to_numeric(df['Distance Covered'], errors='coerce')
        
        # For rows with duration data, calculate speed
        df.loc[mask_too_low | mask_too_high | mask_na, 'Avg_Speed_kmh'] = 50  # Default reasonable speed
        
        # Try to estimate speed from distance and time if available
        for idx, row in df[mask_too_low | mask_too_high | mask_na].iterrows():
            if pd.notna(row['Trip Started At']) and pd.notna(row['Trip Completed At']) and pd.notna(distance_col[idx]):
                try:
                    start_time = pd.to_datetime(row['Trip Started At'])
                    end_time = pd.to_datetime(row['Trip Completed At'])
                    duration_hours = (end_time - start_time).total_seconds() / 3600
                    if duration_hours > 0:
                        calculated_speed = distance_col[idx] / duration_hours
                        # Apply sanity check to calculated speed
                        if 10 <= calculated_speed <= 120:
                            df.loc[idx, 'Avg_Speed_kmh'] = calculated_speed
                except:
                    pass  # Keep default if calculation fails
    except Exception as e:
        print(f"Error processing average speed: {e}")
        df['Avg_Speed_kmh'] = 50  # Default if all else fails
    
    # Define speed efficiency function based on Indian driving conditions
    def speed_efficiency(speed):
        if speed < 20:
            return 1.4  # 40% higher emissions at very low speeds (congestion, stop-and-go)
        elif 20 <= speed < 40:
            return 1.2  # 20% higher at low speeds
        elif 40 <= speed < 60:
            return 1.05  # 5% higher at moderate speeds
        elif 60 <= speed <= 80:
            return 1.0  # Optimal efficiency range for HDVs
        elif 80 < speed <= 100:
            return 1.1  # 10% higher at higher speeds
        else:
            return 1.25  # 25% higher at very high speeds (>100km/h)
    
    df['Speed_Factor'] = df['Avg_Speed_kmh'].apply(speed_efficiency)
    
    # Calculate halt time in hours where available
    # More sophisticated halt time calculation with better edge case handling
    df['Halt_Hours'] = 0.0  # Initialize default value
    
    try:
        # First approach: Use Total Halt Time column if available
        if 'Total Halt Time' in df.columns:
            halt_time = pd.to_numeric(df['Total Halt Time'], errors='coerce')
            
            # Check format - some entries might be in minutes, some might be in hours
            # If mostly large numbers (>100), likely minutes; if mostly small, likely hours
            if halt_time.median() > 100:  # Likely in minutes
                df['Halt_Hours'] = halt_time.fillna(0) / 60
            else:
                df['Halt_Hours'] = halt_time.fillna(0)
                
        # Second approach: Calculate from timestamps if available
        if df['Halt_Hours'].sum() == 0:  # If first approach failed
            for idx, row in df.iterrows():
                halt_hours = 0
                
                # Check if we have left source and reached destination timestamps
                if pd.notna(row.get('Left Source At')) and pd.notna(row.get('Reached At')):
                    try:
                        source_time = pd.to_datetime(row['Left Source At'])
                        dest_time = pd.to_datetime(row['Reached At'])
                        total_trip_hours = (dest_time - source_time).total_seconds() / 3600
                        distance = row['Distance Covered'] if pd.notna(row['Distance Covered']) else 0
                        
                        # Estimate driving time based on distance and average speed
                        avg_speed = row['Avg_Speed_kmh'] if pd.notna(row['Avg_Speed_kmh']) else 50
                        driving_hours = distance / avg_speed
                        
                        # Halt time is total time minus driving time
                        if total_trip_hours > driving_hours:
                            halt_hours = total_trip_hours - driving_hours
                            
                            # Sanity check: halt shouldn't be more than 70% of total trip time
                            if halt_hours > total_trip_hours * 0.7:
                                halt_hours = total_trip_hours * 0.4  # More realistic estimate
                        
                        df.loc[idx, 'Halt_Hours'] = halt_hours
                    except:
                        pass
        
        # Third approach: Use a reasonable estimate based on distance
        # For routes without specific halt data
        mask_missing_halt = df['Halt_Hours'] == 0
        distance = pd.to_numeric(df.loc[mask_missing_halt, 'Distance Covered'], errors='coerce')
        
        # Formula: longer trips have more halts, but not linearly
        # Typical 4-hour rest per 250km driving distance (regulatory requirement in many regions)
        df.loc[mask_missing_halt, 'Halt_Hours'] = distance.fillna(0).apply(lambda x: min(0.5 + x*0.016, x*0.08))
        
    except Exception as e:
        print(f"Error calculating halt hours: {e}")
        # Fallback to simple distance-based estimate
        df['Halt_Hours'] = df['Distance Covered'].fillna(0) * 0.02  # 2% of distance in km = halt hours
    
    # Calculate idling emissions with vehicle type consideration
    # Different vehicles have different idling consumption
    
    # Define idling emission factors based on vehicle size
    def get_idling_factor(vehicle_no):
        """Get appropriate idling emission factor based on vehicle number"""
        if pd.isna(vehicle_no):
            return IDLING_EMISSION_FACTOR  # Default
            
        vehicle_str = str(vehicle_no).lower()
        # Try to infer vehicle size from registration number patterns
        if any(x in vehicle_str for x in ['car', 'small']):
            return 1.6  # Small vehicle - 1.6 kg CO2/hour idle
        elif any(x in vehicle_str for x in ['lcv', 'pick']):
            return 2.1  # Light commercial - 2.1 kg CO2/hour idle
        elif any(x in vehicle_str for x in ['med', 'bus']):
            return 2.8  # Medium vehicle - 2.8 kg CO2/hour idle
        elif any(x in vehicle_str for x in ['trail', 'artic', 'heavy']):
            return 3.5  # Heavy articulated - 3.5 kg CO2/hour idle
        else:
            return IDLING_EMISSION_FACTOR  # Default from HDV standards
    
    # Apply vehicle-specific idling factors
    df['Idling_Factor'] = df['Current Vehicle No.'].apply(get_idling_factor)
    df['Idling_Emissions_kg'] = df['Halt_Hours'] * df['Idling_Factor']
    
    # Calculate delay impact
    df['Delay_Factor'] = 1.0  # Default
    df.loc[df['Transit Status'] == 'Delayed', 'Delay_Factor'] = 1.2  # 20% inefficiency due to delays
    
    # Calculate total emissions
    df['Total_Emissions_kg'] = (df['Base_Emissions_kg'] * df['Weight_Factor'] * 
                              df['Speed_Factor'] * df['Delay_Factor']) + df['Idling_Emissions_kg']
    
    # Calculate emissions that could have been saved if delivered on time
    df['Potential_Savings_kg'] = df['Total_Emissions_kg'] - (df['Base_Emissions_kg'] * df['Weight_Factor'] * df['Speed_Factor'])
    df.loc[df['Transit Status'] != 'Delayed', 'Potential_Savings_kg'] = 0
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_default_data')
def check_default_data():
    """Check if default data file exists"""
    default_file = 'PRL-GreenkoReport-24-25.csv'
    return jsonify({
        'default_data_available': os.path.isfile(default_file)
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    global current_data_file
    
    # Check if file part exists in the request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request'})
        
    file = request.files['file']
    
    # Check if file was actually selected
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
        
    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Set the current data file
        current_data_file = file_path
        
        # Try to load the data to validate it
        try:
            df = load_data()
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': f"Error processing file: {str(e)}"})
    else:
        return jsonify({'success': False, 'error': 'File type not allowed'})

@app.route('/emissions_data')
def emissions_data():
    df = load_data()
    
    # Summary by vehicle
    vehicle_summary = df.groupby('Current Vehicle No.').agg({
        'Total_Emissions_kg': 'sum',
        'Distance Covered': 'sum',
        'Estimated_Weight_Tons': 'mean',
        'Assignment UID': 'count',
        'Emission_Intensity': 'mean'
    }).reset_index()
    vehicle_summary.columns = ['Vehicle', 'Total_Emissions_kg', 'Total_Distance_km', 'Avg_Load_Tons', 'Trip_Count', 'Avg_Emission_Intensity']
    vehicle_summary['Emissions_per_km'] = vehicle_summary['Total_Emissions_kg'] / vehicle_summary['Total_Distance_km']
    
    # Summary by consignee
    consignee_summary = df.groupby('Consignee').agg({
        'Total_Emissions_kg': 'sum',
        'Distance Covered': 'sum',
        'Assignment UID': 'count',
        'Emission_Intensity': 'mean'
    }).reset_index()
    consignee_summary.columns = ['Consignee', 'Total_Emissions_kg', 'Total_Distance_km', 'Trip_Count', 'Avg_Emission_Intensity']
    
    # On-time vs delayed comparison
    delivery_status = df.groupby('Transit Status').agg({
        'Total_Emissions_kg': 'sum',
        'Distance Covered': 'sum',
        'Potential_Savings_kg': 'sum',
        'Assignment UID': 'count',
        'Emission_Intensity': 'mean'
    }).reset_index()
    
    # Time series data: By month
    time_series_monthly = df.groupby('Trip_Month').agg({
        'Total_Emissions_kg': 'sum',
        'Distance Covered': 'sum',
        'Assignment UID': 'count',
        'Potential_Savings_kg': 'sum',
        'Emission_Intensity': 'mean'
    }).reset_index()
    time_series_monthly.columns = ['Month', 'Total_Emissions_kg', 'Total_Distance_km', 'Trip_Count', 'Potential_Savings_kg', 'Avg_Emission_Intensity']
    time_series_monthly['Emissions_per_km'] = time_series_monthly['Total_Emissions_kg'] / time_series_monthly['Total_Distance_km']
    time_series_monthly = time_series_monthly.sort_values('Month')
    
    # Time series data: By week
    time_series_weekly = df.groupby('Trip_Week').agg({
        'Total_Emissions_kg': 'sum',
        'Distance Covered': 'sum',
        'Assignment UID': 'count',
        'Potential_Savings_kg': 'sum'
    }).reset_index()
    time_series_weekly.columns = ['Week', 'Total_Emissions_kg', 'Total_Distance_km', 'Trip_Count', 'Potential_Savings_kg']
    time_series_weekly['Emissions_per_km'] = time_series_weekly['Total_Emissions_kg'] / time_series_weekly['Total_Distance_km']
    time_series_weekly = time_series_weekly.sort_values('Week')
    
    # Trip details with enhanced data for better analytics
    trip_details = df[['Assignment UID', 'Consignment', 'Source', 'Destination', 
                     'Total_Emissions_kg', 'Distance Covered', 'Transit Status', 
                     'Potential_Savings_kg', 'Current Vehicle No.', 'Emission_Intensity',
                     'Trip_Date', 'Estimated_Weight_Tons', 'Speed_Factor', 
                     'Idling_Emissions_kg']].to_dict('records')
    
    # Enhanced KPIs
    emission_intensity = df['Total_Emissions_kg'].sum() / (df['Distance Covered'].sum() * df['Estimated_Weight_Tons'].mean())
    delivery_efficiency = df['Delivery_Efficiency'].mean()
    emission_per_trip = df['Total_Emissions_kg'].sum() / len(df)
    
    result = {
        'vehicle_summary': vehicle_summary.to_dict('records'),
        'consignee_summary': consignee_summary.to_dict('records'),
        'delivery_status': delivery_status.to_dict('records'),
        'time_series_monthly': time_series_monthly.to_dict('records'),
        'time_series_weekly': time_series_weekly.to_dict('records'),
        'trip_details': trip_details,
        'total_emissions': float(df['Total_Emissions_kg'].sum()),
        'total_distance': float(df['Distance Covered'].sum()),
        'total_potential_savings': float(df['Potential_Savings_kg'].sum()),
        'delayed_trips_count': int(df[df['Transit Status'] == 'Delayed']['Assignment UID'].count()),
        'total_trips_count': int(df['Assignment UID'].count()),
        'emission_intensity': float(emission_intensity),
        'delivery_efficiency': float(delivery_efficiency),
        'emission_per_trip': float(emission_per_trip)
    }
    
    return json.dumps(result)

@app.route('/trip_details/<assignment_uid>')
def trip_details(assignment_uid):
    try:
        df = load_data()
        trip_data = df[df['Assignment UID'] == assignment_uid]
        
        if trip_data.empty:
            return json.dumps({"error": f"Trip with Assignment UID {assignment_uid} not found"}), 404
            
        trip = trip_data.iloc[0]
        
        # Convert values safely
        def safe_float(val):
            try:
                return float(val) if not pd.isna(val) else 0.0
            except:
                return 0.0
        
        result = {
            'assignment_uid': assignment_uid,
            'vehicle': str(trip.get('Current Vehicle No.', 'Unknown')),
            'source': str(trip.get('Source', 'Unknown')),
            'destination': str(trip.get('Destination', 'Unknown')),
            'consignment': str(trip.get('Consignment', 'Unknown')),
            'estimated_weight': safe_float(trip['Estimated_Weight_Tons']),
            'distance_covered': safe_float(trip['Distance Covered']),
            'total_emissions': safe_float(trip['Total_Emissions_kg']),
            'base_emissions': safe_float(trip['Base_Emissions_kg']),
            'idling_emissions': safe_float(trip['Idling_Emissions_kg']),
            'transit_status': str(trip.get('Transit Status', 'Unknown')),
            'potential_savings': safe_float(trip['Potential_Savings_kg']),
            'lr_number': str(trip.get('LR No', 'Unknown')),
            'lr_date': str(trip.get('LR Date', 'Unknown')),
            'consignor': str(trip.get('Consignor', 'Unknown')),
            'consignee': str(trip.get('Consignee', 'Unknown'))
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)}), 500
    
    return json.dumps(result)

@app.route('/export_data')
def export_data():
    """Export emissions data as CSV"""
    try:
        df = load_data()
        
        # Select relevant columns for export
        export_columns = [
            'Assignment UID', 'LR No', 'LR Date', 'Current Vehicle No.',
            'Source', 'Destination', 'Consignment', 'Consignor', 'Consignee',
            'Distance Covered', 'Estimated_Weight_Tons', 'Total_Emissions_kg',
            'Base_Emissions_kg', 'Idling_Emissions_kg', 'Potential_Savings_kg',
            'Transit Status', 'Emission_Intensity', 'Trip_Date'
        ]
        
        # Filter columns that exist in the dataframe
        export_df = df[[col for col in export_columns if col in df.columns]]
        
        # Create a temporary CSV file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'emissions_export.csv')
        export_df.to_csv(temp_path, index=False)
        
        # Return the file as a download
        return send_file(temp_path, as_attachment=True, download_name='transport_emissions_report.csv')
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_summary')
def export_summary():
    """Export emissions summary as CSV"""
    try:
        df = load_data()
        
        # Vehicle summary
        vehicle_summary = df.groupby('Current Vehicle No.').agg({
            'Total_Emissions_kg': 'sum',
            'Distance Covered': 'sum',
            'Estimated_Weight_Tons': 'mean',
            'Assignment UID': 'count'
        }).reset_index()
        vehicle_summary.columns = ['Vehicle', 'Total_Emissions_kg', 'Total_Distance_km', 'Avg_Load_Tons', 'Trip_Count']
        vehicle_summary['Emissions_per_km'] = vehicle_summary['Total_Emissions_kg'] / vehicle_summary['Total_Distance_km']
        
        # Create a temporary CSV file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'emissions_summary.csv')
        vehicle_summary.to_csv(temp_path, index=False)
        
        # Return the file as a download
        return send_file(temp_path, as_attachment=True, download_name='emissions_summary_report.csv')
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dashboard')
def dashboard():
    return render_template('bi_dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
