from flask import Flask, request, jsonify
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
import os
from dotenv import load_dotenv
from bson import ObjectId
from datetime import datetime

# Load biến môi trường
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Khởi tạo Flask và MongoDB
app = Flask(__name__)
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Load model
MODEL_PATH = "models/best_model.keras"
model = load_model(MODEL_PATH)

EMOTIONS = ["Anger", "Happiness", "Sadness", "Neutral", "Surprise"]

# Hàm để chuyển đối tượng ObjectId thành chuỗi
def objectid_to_str(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not serializable")

# Hàm để chuyển đổi tất cả ObjectId trong dict thành chuỗi
def serialize_mongo_data(data):
    if isinstance(data, dict):
        return {k: serialize_mongo_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_mongo_data(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filepath = os.path.join("static/images", file.filename)
    file.save(filepath)

    # Tiền xử lý ảnh
    image = preprocess_image(filepath)

    # Dự đoán từ mô hình
    predictions = model.predict(image)

    # Tìm nhãn có xác suất cao nhất
    emotion_idx = predictions.argmax()
    emotion = EMOTIONS[emotion_idx]

    # Lưu kết quả vào MongoDB
    prediction_data = {
        "filename": file.filename,
        "emotion": emotion,
        "confidence": float(predictions[0][emotion_idx]),
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(prediction_data)

    # Chuyển đổi ObjectId thành chuỗi nếu có
    prediction_data = serialize_mongo_data(prediction_data)

    return jsonify(prediction_data), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Lấy cổng từ biến môi trường
    app.run(debug=True, host='0.0.0.0', port=port)
