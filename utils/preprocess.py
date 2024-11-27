import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(48, 48)):
    """Tiền xử lý ảnh xám (grayscale), bao gồm phát hiện khuôn mặt và chuẩn hóa ảnh"""
    
    # Đảm bảo đường dẫn tệp Cascade Classifier là hợp lệ
    face_cascade_path = 'models/haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"Face cascade file not found at {face_cascade_path}")
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Chuyển ảnh sang grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    
    # Cắt khuôn mặt đầu tiên được phát hiện
    x, y, w, h = faces[0]
    face = image_gray[y:y+h, x:x+w]
    
    # Resize ảnh khuôn mặt về kích thước mục tiêu
    face = cv2.resize(face, target_size)
    
    # Chuẩn hóa ảnh
    face = face.astype("float32") / 255.0
    
    # Trả về ảnh đã chuẩn hóa và thêm chiều batch
    return np.expand_dims(face, axis=(0, -1))
