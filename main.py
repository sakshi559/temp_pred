import os
import cv2
from sklearn.model_selection import train_test_split
from model import load_images, create_model, train_model, evaluate_model, predict_temperature

# Define the paths to your dataset folders
image_folder = 'C:\\pythonProject1\\pythonProject1\\pcb_temp_mngmt\\infrared\\train'  # Replace with the path to your image folder

# Define the temperature values and annotations directly in the code
temperatures = [25.4, 26.8, 24.9, 27.2, 25.1]  # Replace with your temperature values
annotations = [[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70], [50, 60, 70, 80]]  # Replace with your annotations

# Preprocess the images and annotations
desired_width = 100  # Desired image width for preprocessing
desired_height = 100  # Desired image height for preprocessing
original_width = 250
original_height = 250

processed_images, processed_annotations = load_images(image_folder, annotations, desired_width, desired_height, original_width, original_height)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_images, temperatures, test_size=0.2, random_state=42)

# Create the model
model = create_model(desired_height, desired_width)

# Train the model
batch_size = 32
epochs = 10
train_model(model, X_train, list(y_train), X_test, list(y_test), batch_size, epochs)


# Evaluate the model
evaluate_model(model, X_test, list(y_test))

# Make predictions on new, unseen data
new_image_path = 'C:\\pythonProject1\\pythonProject1\\pcb_temp_mngmt\\infrared\\train\\image8.jpg'  # Path to the new image
predict_temperature(model, new_image_path, desired_width, desired_height, original_width, original_height)
