import cog
import os
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
from datetime import datetime

# Load model
MODEL_PATH = "models/best_model.keras"
model = load_model(MODEL_PATH)

MODEL_PATH2 = "models/best_model1.keras"
model2 = load_model(MODEL_PATH2)

EMOTIONS = ["Anger", "Happiness", "Sadness", "Neutral", "Surprise"]
EMOTIONS2 = ["Anger", "Happiness", "Sadness", "Neutral"]

class Predictor(cog.Predictor):
    def setup(self):
        """Khởi tạo mô hình"""
        self.model = load_model(MODEL_PATH)
        self.model2 = load_model(MODEL_PATH2)

    def predict(self, image):
        """Dự đoán cảm xúc từ ảnh"""
        # Tiền xử lý ảnh
        image = preprocess_image(image)

        # Dự đoán từ cả hai model
        predictions = self.model.predict(image)
        predictions2 = self.model2.predict(image)

        # Tìm nhãn có xác suất cao nhất từ cả hai model
        emotion_idx = predictions.argmax()
        emotion = EMOTIONS[emotion_idx]

        emotion_idx2 = predictions2.argmax()
        emotion2 = EMOTIONS2[emotion_idx2]
        
        if emotion == emotion2:
            final_emotion = emotion  # Nếu cả hai model đều dự đoán cùng một nhãn
        else:
            final_emotion = emotion  # Hoặc bạn có thể chọn từ mô hình có độ tin cậy cao hơn

        return {
            "emotion": final_emotion,
            "confidence": float(predictions[0][emotion_idx])
        }
