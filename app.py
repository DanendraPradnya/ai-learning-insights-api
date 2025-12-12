from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_assets', 'rf_model.joblib')
ENCODER_PATH = os.path.join(BASE_DIR, 'model_assets', 'label_encoder.joblib')

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

def extract_features_from_json(payload):
  trackings = payload.get('trackings', [])
  completions = payload.get('completions', [])

  avg_materials_per_days = 0.0
  std_daily_materials = 0.0
  active_days = 0.0
  avg_enrolling_times = 0.0
  total_completions = 0.0

  if trackings:
    df = pd.DataFrame(trackings)
    if 'last_viewed' in df.columns:
      df['date'] = pd.to_datetime(df['last_viewed']).dt.date
      daily_counts = df.groupby('date').size()
          
      avg_materials_per_days = daily_counts.mean()
      std_daily_materials = daily_counts.std() if len(daily_counts) > 1 else 0.0
      active_days = len(daily_counts)

  if completions:
    df_c = pd.DataFrame(completions)
    if 'enrolling_times' in df_c.columns:
      avg_enrolling_times = df_c['enrolling_times'].mean()
    total_completions = len(df_c)

  return np.array([[avg_materials_per_days, std_daily_materials, active_days, avg_enrolling_times, total_completions]])

@app.route('/predict', methods=['POST'])
@cross_origin(origins="*")
def predict():
    try:
      data = request.json
        
      features = extract_features_from_json(data)
        
      prediction_index = model.predict(features)[0]
      prediction_label = encoder.inverse_transform([prediction_index])[0]
        
      return jsonify({
        'status': 'success',
        'learning_style': prediction_label,
        'features_calculated': {
            'avg_materials_per_days': features[0][0],
            'std_daily_materials': features[0][1],
            'avg_enrolling_times': features[0][3]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port, debug=True)