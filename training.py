import os
import cv2
import dlib
import numpy as np
import pickle

# Đường dẫn đến thư mục chứa dataset
dataset_path = "Dataset/"

# Khởi tạo detector và face descriptor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Tải mô hình landmarks
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Tải mô hình nhận diện khuôn mặt

known_face_encodings = []
known_face_names = []

# Lặp qua từng thư mục trong dataset
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)

            # Phát hiện khuôn mặt
            dets = detector(image, 1)
            for d in dets:
                # Lấy landmarks
                shape = predictor(image, d)
                # Tính encoding khuôn mặt
                face_encoding = np.array(face_rec_model.compute_face_descriptor(image, shape))
                
                known_face_encodings.append(face_encoding)  # Lưu encoding khuôn mặt
                known_face_names.append(person_name)

# Lưu mô hình vào file
with open('face_encodings.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Huấn luyện hoàn tất!")