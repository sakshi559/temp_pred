import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Define the paths to your dataset folders
image_folder = 'C:\pythonProject1\pythonProject1\pcb_temp_mngmt\infrared\train'  # Replace 'dataset/images' with the path to your image folder

# Define the temperature values and annotations directly in the code
temperatures = [106.2, 80.6, 34.7, 37.3, 61.2, 
  52.6, 58.8, 58.1, 37.3, 58.8,
  65.3, 98.2, 36.7, 37.8, 106.7, 
  37.2, 40.5, 105, 27.5, 25.3,
  41.5, 51.6, 34.7, 71.9, 40.8, 
  39.0, 39.1, 76.0]  # Replace with your temperature values
annotations = [[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70], [50, 60, 70, 80]]  # Replace with your annotations

# Load the image paths from the image folder
image_paths = []
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust the file extensions as per your dataset
        image_paths.append(os.path.join(image_folder, filename))

# Preprocess the images and annotations
processed_images = []
processed_annotations = []

desired_width=224
desired_height=224

for image_path, annotation in zip(image_paths, annotations):
    # Load the image
    image = cv2.imread(image_path, 0)  # Load as grayscale image using OpenCV

    # Preprocess the image (e.g., resize, normalize, etc.)
    image = cv2.resize(image, (desired_width, desired_height))  # Resize to desired dimensions
    image = image.astype('float32') / 255.0  # Normalize pixel values between 0 and 1

    # Preprocess the bounding box coordinates (if necessary)
    processed_annotation = [annotation[0] / original_width,  # Normalize x-coordinate
                            annotation[1] / original_height,  # Normalize y-coordinate
                            annotation[2] / original_width,  # Normalize width
                            annotation[3] / original_height]  # Normalize height

    # Append the preprocessed data to the lists
    processed_images.append(image)
    processed_annotations.append(processed_annotation)

# Convert the lists to NumPy arrays
processed_images = np.array(processed_images)
processed_annotations = np.array(processed_annotations)
temperatures = np.array(temperatures)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, train_boxes, test_boxes = train_test_split(processed_images,
                                                                            temperatures,
                                                                            processed_annotations,
                                                                            test_size=0.2,
                                                                            random_state=42)
