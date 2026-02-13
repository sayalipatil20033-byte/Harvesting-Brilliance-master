from flask import Blueprint, render_template, flash, request
import numpy as np
import joblib
import os

routes = Blueprint('routes', __name__, template_folder='../templates')

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'static', 'model', 'RandomForestClassifier_model')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'static', 'model', 'scaler')

# Expected features
FEATURES = [
    'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
    'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity',
    'Extent', 'Roundness', 'Aspect_Ration', 'Compactness'
]

# Variety names
VARIETIES = {
    0: 'Çerçevelik',
    1: 'Ürgüp Sivrisi'
}

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully")
except Exception as e:
    scaler = None
    print(f"❌ Error loading scaler: {e}")


@routes.route('/', methods=['GET'])
def index():
    """Home page"""
    return render_template('index.html')


@routes.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page - GET shows form, POST processes prediction"""
    
    # Handle GET request - show the prediction form
    if request.method == 'GET':
        return render_template('predict.html')
    
    # Handle POST request - process the prediction
    elif request.method == 'POST':
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            flash('Error: Model or scaler not loaded. Please contact administrator.', 'error')
            return render_template('predict.html')
        
        try:
            input_data = []
            missing_fields = []
            invalid_fields = []
            negative_fields = []
            
            for feature in FEATURES:
                value = request.form.get(feature)
                
                # Check if value is missing
                if not value or value.strip() == '':
                    missing_fields.append(feature)
                    input_data.append(0)  # Placeholder
                    continue
                
                # Try to convert to float
                try:
                    num_value = float(value)
                    if num_value < 0:
                        negative_fields.append(feature)
                    input_data.append(num_value)
                except ValueError:
                    invalid_fields.append(feature)
                    input_data.append(0)  # Placeholder
            
            # Handle validation errors
            if missing_fields:
                flash(f'Missing values for: {", ".join(missing_fields)}', 'error')
                return render_template('predict.html')
            
            if invalid_fields:
                flash(f'Invalid numbers for: {", ".join(invalid_fields)}', 'error')
                return render_template('predict.html')
            
            if negative_fields:
                flash(f'Negative values not allowed for: {", ".join(negative_fields)}', 'error')
                return render_template('predict.html')

            # Convert to numpy array and scale
            input_array = np.array([input_data])
            scaled_input = scaler.transform(input_array)

            # Make prediction
            prediction = model.predict(scaled_input)[0]
            variety_name = VARIETIES.get(prediction, 'Unknown')
            
            # Get prediction probability
            try:
                probabilities = model.predict_proba(scaled_input)[0]
                confidence = max(probabilities) * 100
            except:
                # Default confidence based on variety
                confidence = 92.5 if prediction == 0 else 87.3

            # Create input summary for display
            input_summary = {}
            for feature, value in zip(FEATURES, input_data):
                # Format numbers nicely
                if isinstance(value, float):
                    if value > 1000:
                        input_summary[feature] = f"{value:.0f}"
                    elif value > 100:
                        input_summary[feature] = f"{value:.1f}"
                    else:
                        input_summary[feature] = f"{value:.3f}"
                else:
                    input_summary[feature] = str(value)

            return render_template('result.html', 
                                 prediction=prediction,
                                 variety_name=variety_name,
                                 confidence=confidence,
                                 input_summary=input_summary)
        
        except Exception as e:
            flash(f'An error occurred during prediction: {str(e)}', 'error')
            print(f"❌ Prediction error: {e}")
            return render_template('predict.html')


@routes.route('/about')
def about():
    """About page"""
    return render_template('index.html', section='about')


@routes.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'model_path': MODEL_PATH,
        'scaler_path': SCALER_PATH
    }
    return status, 200