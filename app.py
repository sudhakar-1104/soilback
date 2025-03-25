from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json

app = Flask(__name__)
CORS(app)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

# Route to check if server is running
@app.route('/')
def home():
    return jsonify({"message": "🌱 Crop Yield Prediction API is running!"})

# Helper function to run Python scripts for predictions
def run_python_script(script_name, input_data, model_path, encoders_path):
    try:
        # Prepare command
        command = [
            'python',
            os.path.join(SCRIPTS_DIR, script_name),
            json.dumps(input_data),
            model_path,
            encoders_path
        ]
        # Run the script
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr.strip()}

# Crop yield prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_crop_yield():
    try:
        data = request.json
        # Validate input data
        required_fields = ['area', 'cropType', 'year', 'averageRainfall', 'avgTemp', 'pesticides']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Prepare input data
        input_data = {
            "area": data['area'],
            "cropType": int(data['cropType']),
            "year": int(data['year']),
            "averageRainfall": float(data['averageRainfall']),
            "avgTemp": float(data['avgTemp']),
            "pesticides": float(data['pesticides'])
        }

        # Run prediction script
        result = run_python_script(
            'predict.py',
            input_data,
            os.path.join(MODELS_DIR, 'crop_yield_predictor.pkl'),
            os.path.join(MODELS_DIR, 'label_encoders.pkl')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Soil health prediction endpoint
@app.route('/soil_predict', methods=['POST'])
def predict_soil_health():
    try:
        data = request.json
        # Validate input data
        required_fields = ['sand', 'clay', 'silt', 'pH', 'EC']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Prepare input data
        input_data = {
            "sand": int(data['sand']),
            "clay": int(data['clay']),
            "silt": float(data['silt']),
            "pH": float(data['pH']),
            "EC": float(data['EC'])
        }

        # Run prediction script
        result = run_python_script(
            'predict_2.py',
            input_data,
            os.path.join(MODELS_DIR, 'soil_health_model.pkl'),
            os.path.join(MODELS_DIR, 'label_encoders.pkl')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(port=80, debug=True)
