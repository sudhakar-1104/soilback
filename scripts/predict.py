#!/usr/bin/env python3
"""
predict.py - Helper script to load and use pickle models in Node.js environment
Usage: python predict.py <json_input_data> <model_path> <encoders_path>
"""

import sys
import json
import joblib
import numpy as np

def predict_crop_yield(input_data, model_path, encoders_path):
    try:
        # Load model and label encoders
        crop_yield_model = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)
        
        # Extract features
        area = input_data['area']
        crop_type = input_data['cropType']
        year = input_data['year']
        rainfall = input_data['averageRainfall']
        avg_temp = input_data['avgTemp']
        pesticides = input_data['pesticides']
        
        # Handle area encoding
        if area in label_encoders["Area"].classes_:
            area_encoded = label_encoders["Area"].transform([area])[0]
        else:
            print(f"Warning: Unseen area '{area}' found. Using default encoding.", file=sys.stderr)
            area_encoded = -1  # Use -1 for unknown areas
        
        # Prepare input features
        input_array = np.array([[area_encoded, crop_type, year, rainfall, avg_temp, pesticides]])
        
        # Make prediction
        prediction = float(crop_yield_model.predict(input_array)[0])
        
        return {"prediction": round(prediction, 2)}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) != 4:
        print("Usage: python predict.py <json_input_data> <model_path> <encoders_path>", file=sys.stderr)
        sys.exit(1)
    
    input_json = sys.argv[1]
    model_path = sys.argv[2]
    encoders_path = sys.argv[3]
    
    # Parse input JSON
    input_data = json.loads(input_json)
    
    # Make prediction
    result = predict_crop_yield(input_data, model_path, encoders_path)
    
    # Output result as JSON
    print(json.dumps(result))