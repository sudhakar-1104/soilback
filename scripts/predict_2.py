#!/usr/bin/env python3
"""
predict.py - Helper script to load and use pickle models in Node.js environment
Usage: python predict.py <json_input_data> <model_path> <encoders_path>
"""

import sys
import json
import joblib
import numpy as np

def predict_soil_health(input_data, model_path, encoders_path):
    try:
        # Load model and label encoders
        soil_yield_model = joblib.load(model_path)
       
        
        # Extract features
        sand = input_data['sand']
        clay = input_data['clay']
        silt = input_data['silt']
        pH = input_data['pH']
        EC = input_data['EC']
   
        # Prepare input features
        input_array = np.array([[sand, clay, silt, pH, EC]])
        
        # Make prediction
        prediction = float(soil_yield_model.predict(input_array)[0])
        
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
    result = predict_soil_health(input_data, model_path, encoders_path)
    
    # Output result as JSON
    print(json.dumps(result))