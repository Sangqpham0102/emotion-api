from flask import Flask, request, jsonify
from pymongo import MongoClient
from tensorflow.keras.models import load_model
import io
import os
from dotenv import load_dotenv
from bson import ObjectId
from datetime import datetime
from utils.preprocess import preprocess_image  # Đảm bảo preprocess_image có sẵn

# Tải các biến môi trường từ file .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Khởi tạo Flask và MongoDB
app = Flask(__name__)
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Tải mô hình khi ứng dụng bắt đầu
MODEL_PATH = "models/best_model.keras"
model = load_model(MODEL_PATH)

EMOTIONS = ["Anger", "Happiness", "Sadness", "Neutral", "Surprise"]

# Hàm chuyển ObjectId thành chuỗi để lưu vào MongoDB
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
    # Chuyển file ảnh thành io.BytesIO trước khi gửi vào preprocess
    image = preprocess_image(io.BytesIO(file.read()))

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

# Cấu hình Flask để chạy trên cổng được chỉ định từ biến môi trường
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Cổng sẽ được Render tự động gán
    app.run(debug=True, host='0.0.0.0', port=port)
