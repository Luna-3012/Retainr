"""
Retainr Churn Prediction API
Flask server that serves ML predictions to Salesforce.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'churn_model.pkl')
model_data = None


def load_model():
    """Load the trained model."""
    global model_data
    try:
        model_data = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully (version: {model_data['model_version']})")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'model_version': model_data['model_version'] if model_data else None,
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict_churn():
    """
    Predict churn probability for an account.
    
    Expected JSON payload:
    {
        "account_id": "001XX000003ABCD",
        "login_frequency": 45,
        "feature_usage_score": 82,
        "support_ticket_count": 1,
        "nps_score": 9,
        "contract_value": 50000,
        "duration_months": 24
    }
    """
    try:
        # Validate model is loaded
        if model_data is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = model_data['feature_columns']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Prepare features (raw — Random Forest does not use a scaler)
        feature_columns = model_data['feature_columns']
        features = pd.DataFrame([[data[f] for f in feature_columns]], columns=feature_columns)

        # Predict
        churn_probability = model_data['model'].predict_proba(features)[0][1]
        churn_score = round(churn_probability * 100, 2)

        # Determine risk level
        if churn_score >= 75:
            risk_level = 'Critical'
        elif churn_score >= 50:
            risk_level = 'High'
        elif churn_score >= 25:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        # Get contributing factors
        contributing_factors = get_contributing_factors(data)
        
        # Build response
        response = {
            'success': True,
            'account_id': data.get('account_id', ''),
            'churn_score': churn_score,
            'risk_level': risk_level,
            'contributing_factors': contributing_factors,
            'model_version': model_data['model_version'],
            'prediction_date': datetime.utcnow().isoformat(),
            'confidence': round(max(churn_probability, 1 - churn_probability) * 100, 2)
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction for multiple accounts.
    
    Expected JSON payload:
    {
        "accounts": [
            {
                "account_id": "001XX000003ABCD",
                "login_frequency": 45,
                ...
            },
            ...
        ]
    }
    """
    try:
        if model_data is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        data = request.get_json()
        
        if not data or 'accounts' not in data:
            return jsonify({
                'success': False,
                'error': 'No accounts data provided'
            }), 400
        
        predictions = []
        
        feature_defaults = {
            'login_frequency': 0,
            'feature_usage_score': 0,
            'support_ticket_count': 0,
            'nps_score': 5,
            'contract_value': 0,
            'duration_months': 0
        }
        feature_columns = model_data['feature_columns']

        for account in data['accounts']:
            features = pd.DataFrame(
                [[account.get(f, feature_defaults[f]) for f in feature_columns]],
                columns=feature_columns
            )

            churn_probability = model_data['model'].predict_proba(features)[0][1]
            churn_score = round(churn_probability * 100, 2)

            if churn_score >= 75:
                risk_level = 'Critical'
            elif churn_score >= 50:
                risk_level = 'High'
            elif churn_score >= 25:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'

            predictions.append({
                'account_id': account.get('account_id', ''),
                'churn_score': churn_score,
                'risk_level': risk_level,
                'contributing_factors': get_contributing_factors(account),
                'model_version': model_data['model_version'],
                'prediction_date': datetime.utcnow().isoformat()
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_processed': len(predictions)
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def get_contributing_factors(data):
    """
    Determine top contributing factors for churn risk.
    Uses feature importance and actual values to explain prediction.
    """
    feature_names = model_data['feature_columns']
    importances = model_data['model'].feature_importances_
    
    # Define thresholds for "bad" values
    thresholds = {
        'login_frequency': {'bad_below': 15, 'label': 'Low Login Frequency'},
        'feature_usage_score': {'bad_below': 40, 'label': 'Low Feature Adoption'},
        'support_ticket_count': {'bad_above': 5, 'label': 'High Support Tickets'},
        'nps_score': {'bad_below': 4, 'label': 'Low NPS Score'},
        'contract_value': {'bad_below': 20000, 'label': 'Low Contract Value'},
        'duration_months': {'bad_below': 6, 'label': 'Short Tenure'}
    }
    
    factors = []
    
    for feature in feature_names:
        value = data.get(feature, 0)
        threshold = thresholds.get(feature, {})
        
        if 'bad_below' in threshold and value < threshold['bad_below']:
            factors.append({
                'factor': threshold['label'],
                'value': value,
                'importance': round(importances[feature_names.index(feature)] * 100, 1)
            })
        elif 'bad_above' in threshold and value > threshold['bad_above']:
            factors.append({
                'factor': threshold['label'],
                'value': value,
                'importance': round(importances[feature_names.index(feature)] * 100, 1)
            })
    
    # Sort by importance
    factors.sort(key=lambda x: x['importance'], reverse=True)
    
    # Return top 3 factors as string
    if factors:
        return '; '.join([f"{f['factor']} ({f['value']})" for f in factors[:3]])
    else:
        return 'No significant risk factors identified'


# Load model on startup
load_model()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)