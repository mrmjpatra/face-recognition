import os
import numpy as np
import cv2
import tensorflow as tf

def load_data(data_dir):
    print("Loading the images from directory...")
    images = []
    labels = []
    label_names = []

    try:
        # Iterate through each person in the directory
        for label, person_name in enumerate(os.listdir(data_dir)):
            person_dir = os.path.join(data_dir, person_name)
            
            # Skip if not a directory
            if not os.path.isdir(person_dir):
                continue

            # Add the person's name to label_names
            label_names.append(person_name)

            # Iterate through each image file in the person's directory
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                
                try:
                    # Load the image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Unable to load image {image_path}")
                        continue
                    
                    # Resize the image to the size expected by the model (160x160)
                    image = cv2.resize(image, (160, 160))
                    
                    # Append the image and label to the respective lists
                    images.append(image)
                    labels.append(label)

                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue

    except Exception as e:
        print(f"Error loading data from directory {data_dir}: {e}")
        return None, None, None

    # Convert lists to numpy arrays
    images = np.array(images, dtype="float32")
    labels = np.array(labels)

    print("Finished loading. Returning images, labels, and label names.")
    return images, labels, label_names

def save_model(model, model_path):
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
