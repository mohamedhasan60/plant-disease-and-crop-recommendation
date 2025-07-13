from flask import Flask, request, send_from_directory, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

# تحميل الموديل و LabelEncoder
model = load_model('climate_plant_model.keras')
le = joblib.load('label_encoder.pkl')

# تحميل البيانات وتطبيعها
data = pd.read_csv('climate_plant_dataset.csv')
X = data[['Temperature', 'Humidity']]
scaler = StandardScaler()
scaler.fit(X)

# قراءة ملف JSON
with open('plant_info.json', 'r', encoding='utf-8') as f:
    plant_info = json.load(f)

def get_plant_details(plant_name):
    plant_name = plant_name.strip()
    details = plant_info.get(plant_name, {
        "details": "No details available",
        "season": "Unknown",
        "planting_period": "Unknown",
        "planting_method": "Unknown",
        "harvesting_method": "Unknown",
        "image_path": "images/default.jpg"
    })
    details["details"] = details["details"].replace("°C", "C")
    return details

@app.route('/')
def home():
    return '✅ API is working. Use /predict_plant?Temperature=25&Humidity=60'

@app.route('/predict_plant', methods=['GET'])
def predict_plant():
    temperature = float(request.args.get('Temperature', 0))
    humidity = float(request.args.get('Humidity', 0))

    input_data = np.array([[temperature, humidity]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    top_n = 3
    top_indices = np.argsort(prediction[0])[::-1][:top_n]
    plant_names = le.inverse_transform(top_indices)

    response = []
    for plant_name in plant_names:
        plant_name = plant_name.strip()
        details = get_plant_details(plant_name)
        image_path = details.get("image_path", "images/default.jpg")
        image_filename = image_path.split('/')[-1]
        image_url = f"https://web-production-856c2.up.railway.app/images/{image_filename}"

        plant_data = {
            "plant_name": plant_name,
            "details": details["details"],
            "season": details["season"],
            "planting_period": details["planting_period"],
            "planting_method": details["planting_method"],
            "harvesting_method": details["harvesting_method"],
            "image_path": image_url
        }
        response.append(plant_data)

    return jsonify(response)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
