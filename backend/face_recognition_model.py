import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import load_data, save_model

def train_and_save_model(data_dir, model_path):
    try:
        # Load data
        images, labels, label_names = load_data(data_dir)
        # Check if data loading was successful
        if images is None or labels is None or label_names is None:
            raise ValueError("Data loading failed.")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Normalize images
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        print("load pre trained model")
        # Load pre-trained model (e.g., FaceNet, ResNet)
        try:
            weights_path = './models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5'
            base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights=None)
            base_model.load_weights(weights_path)
            base_model.trainable = False
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            raise ValueError("Pre-trained model loading failed.")
        print("add custom layers")
        # Add custom layers on top
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(label_names), activation='softmax')  # Output layer with the number of classes
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Data augmentation to improve generalization
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        datagen.fit(X_train)

        # Train the model
        model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=10)

        # Save the trained model
        save_model(model, model_path)
        print(f"Model saved to {model_path}")

    except Exception as e:
        print(f"Error in training model: {e}")

# Example usage:
if __name__ == "__main__":
    data_dir = 'dataset/'
    model_path = './models/face_recognition_model.keras'
    train_and_save_model(data_dir, model_path)
