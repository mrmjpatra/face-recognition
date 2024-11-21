import os
import cv2
import numpy as np
import tensorflow as tf
from utils import load_model, load_data
import face_recognition

def recognize_face_from_image(image_data, model, label_names, threshold=0.6):
    try:
        # Preprocess the image
        image = cv2.resize(image_data, (160, 160)) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict the class
        preds = model.predict(image)
        pred_label = np.argmax(preds)
        confidence = preds[0][pred_label]

        # Check if confidence is above threshold
        if confidence >= threshold:
            return label_names[pred_label]
        else:
            return "Unknown"

    except Exception as e:
        print(f"Error in face recognition: {e}")
        return None

def check_camera_permission():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Unable to access the camera. Please check camera permissions in your system settings.")
        else:
            cap.release()  # Release the camera if opened successfully
            return True
    except Exception as e:
        print(e)
        return False

def recognize_face_from_camera(model_path, label_names, threshold=0.6):
    try:
        # Check if the camera is accessible
        if not check_camera_permission():
            print("Please enable camera permissions in your system settings.")
            return

        # Load the pre-trained model
        model = load_model(model_path)

        # Access the camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("Unable to access the camera")

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert the image to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_frame)

            # Recognize faces in the frame
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = rgb_frame[top:bottom, left:right]
                name = recognize_face_from_image(face_image, model, label_names, threshold)

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw the name below the face
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('Face Recognition', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error in real-time face recognition: {e}")

if __name__ == "__main__":
    data_dir = 'dataset/'  # Path to your dataset folder

    # Load label names dynamically based on folder names
    _, _, label_names = load_data(data_dir)  # This function returns images, labels, and label_names
    
    model_path = './models/face_recognition_model.keras'
    
    recognize_face_from_camera(model_path, label_names)
