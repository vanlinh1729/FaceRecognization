import cv2
import os

# Tạo thư mục để lưu trữ dataset
output_dir = 'Huong'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Số lượng ảnh muốn thu thập
num_images = 100
image_count = 0

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Sử dụng bộ nhận diện khuôn mặt của OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while image_count < num_images:
    # Đọc ảnh từ camera
    ret, frame = cap.read()
    
    if not ret:
        print("Không thể lấy hình ảnh từ camera.")
        break

    # Chuyển ảnh sang grayscale để nhận diện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ khung hình xung quanh khuôn mặt và lưu lại ảnh khuôn mặt
    for (x, y, w, h) in faces:
        # Vẽ khung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cắt khuôn mặt ra khỏi ảnh
        face_img = frame[y:y + h, x:x + w]

        # Lưu khuôn mặt thành file ảnh
        face_filename = os.path.join(output_dir, f"face_{image_count}.jpg")
        cv2.imwrite(face_filename, face_img)
        
        image_count += 1
        print(f"Lưu ảnh: {face_filename}")

    # Hiển thị khung hình với các khuôn mặt nhận diện được
    cv2.imshow('Capturing Faces', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng các cửa sổ
cap.release()
cv2.destroyAllWindows()
