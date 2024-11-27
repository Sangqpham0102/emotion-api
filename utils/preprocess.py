import cv2
import numpy as np
import io

def preprocess_image(image_file, target_size=(48, 48)):
    """Tiền xử lý ảnh xám (grayscale), bao gồm phát hiện khuôn mặt và chuẩn hóa ảnh"""
    # Đọc dữ liệu ảnh từ BytesIO
    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Image could not be read")

    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    
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
