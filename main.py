import cv2
import numpy as np
import pickle
import dlib

# Load the saved face encodings
with open('face_encodings.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Initialize the dlib face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Open the camera
video_capture = cv2.VideoCapture(1)  # Use the default camera (replace with 1 if not working)

if not video_capture.isOpened():
    print("Cannot open camera.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Cannot receive frame from the camera.")
        break

    # Convert the image from BGR (OpenCV default) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector(rgb_frame)

    for face in faces:
        # Get the landmarks/shape of the face
        shape = predictor(rgb_frame, face)
        # Convert the shape to a face descriptor (encoding)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))

        # Compare this face encoding to known face encodings
        matches = [np.linalg.norm(known_encoding - face_encoding) for known_encoding in known_face_encodings]
        name = "Unknown"

        # If a match is found
        if matches:
            best_match_index = np.argmin(matches)
            if matches[best_match_index] < 0.4:  # Matching threshold
                name = known_face_names[best_match_index]

        # Draw a box around the face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for Unknown
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)

        # Add a label with the name
        text_color = (255, 255, 255) if name != "Unknown" else (0, 0, 0)  # White for known, Black for Unknown
        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()
