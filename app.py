from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and encoders
try:
    role_encoder = joblib.load('role_encoder.joblib')  # Encoder for employee_role
    transaction_model = joblib.load('transaction_model.joblib')  # Trained prediction model
    type_encoder = joblib.load('type_encoder.joblib')  # Encoder for type (for inverse_transform)
except FileNotFoundError as e:
    print(f"Error loading .joblib files: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Extract and convert features
        role_idx = role_encoder.transform([data['employee_role']])[0]
        features = [
            float(data['quantity']),
            float(data['transaction_month']),
            float(data['days_since_date']),
            float(role_idx),  # Ensure role_idx is a float if model expects it
            float(data['material_quantity'])
        ]

        # Make prediction
        prediction = transaction_model.predict([features])[0]

        # Decode the prediction to a type
        if hasattr(type_encoder, 'inverse_transform'):  # Use type_encoder if it’s a LabelEncoder
            predicted_type = type_encoder.inverse_transform([prediction])[0]
        else:  # Fallback to manual mapping if type_encoder isn’t suitable
            type_map = {0: 'standard', 1: 'urgent', 2: 'high_value'}  # Adjust based on your training
            predicted_type = type_map.get(prediction, 'unknown')

        return jsonify({'type': predicted_type})

    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid data: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)