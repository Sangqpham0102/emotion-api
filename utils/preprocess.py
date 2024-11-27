import cv2
import numpy as np

def preprocess_image(image_path, target_size=(48, 48)):
    """Tiền xử lý ảnh xám (grayscale), bao gồm phát hiện khuôn mặt và chuẩn hóa ảnh"""
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    x, y, w, h = faces[0]
    face = image_gray[y:y+h, x:x+w]
    face = cv2.resize(face, target_size)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=(0, -1))
