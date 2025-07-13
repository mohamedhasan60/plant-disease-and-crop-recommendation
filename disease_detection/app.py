from flask import Flask, request, send_from_directory, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

model = load_model('plant_disease_model.keras')  

@app.route('/predict_health', methods=['POST'])
def predict_health():
    if 'image' not in request.files and request.content_type != 'image/raw':
        return jsonify({"error": "No image or unsupported format"}), 400

    if request.content_type == 'image/jpeg':
        img_bytes = request.files['image'].read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif request.content_type == 'image/raw':
        img_bytes = request.get_data()
        # تحويل RGB565 إلى BGR
        img_array = np.frombuffer(img_bytes, dtype=np.uint16)
        height, width = 480, 640  # حسب frame_size (VGA)
        img_array = img_array.reshape((height, width))
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[..., 0] = ((img_array & 0x001F) << 3)  # Blue
        img[..., 1] = ((img_array & 0x07E0) >> 3)  # Green
        img[..., 2] = ((img_array & 0xF800) >> 8)  # Red
        success, encoded_img = cv2.imencode('.jpg', img)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    else:
        return jsonify({"error": "Unsupported content type"}), 400

    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # تحويل الصورة لتناسب الموديل
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # التنبؤ
    prediction = model.predict(img)
    result = "صحيح" if prediction[0][0] > 0.5 else "غير صحيح"

    response = {
        "status": result,
        "confidence": float(prediction[0][0])
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)