from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tensorflow as tf
import face_recognition
from utils import load_model, load_data
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Load model and label names
MODEL_PATH = './models/face_recognition_model.keras'
DATA_DIR = 'dataset/'
model = load_model(MODEL_PATH)
_, _, label_names = load_data(DATA_DIR)

def recognize_faces_from_frame(frame, model, label_names, threshold=0.6):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    recognized_faces = []

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = rgb_frame[top:bottom, left:right]
        try:
            resized_face = cv2.resize(face_image, (160, 160)) / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)
            preds = model.predict(resized_face)
            pred_label = np.argmax(preds)
            confidence = preds[0][pred_label]

            name = label_names[pred_label] if confidence >= threshold else "Unknown"
            recognized_faces.append({"name": name, "location": [top, right, bottom, left]})
        except Exception as e:
            print(f"Error processing face: {e}")
    return recognized_faces

@app.route('/')
def index():
    return jsonify({"message": "Flask app is running!"})

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Decode base64 image data
        img_data = base64.b64decode(data['frame'])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Recognize faces
        recognized_faces = recognize_faces_from_frame(frame, model, label_names)
        print(recognized_faces,'recognized_faces')
        emit('recognized_faces', {'faces': recognized_faces})
    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True)
