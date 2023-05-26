import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LambdaCallback


def load_images(image_folder, annotations, desired_width, desired_height, original_width, original_height):
    image_paths = []
    processed_images = []
    processed_annotations = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_paths.append(os.path.join(image_folder, filename))

    for image_path, annotation in zip(image_paths, annotations):
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (desired_width, desired_height))
        image = image.astype('float32') / 255.0

        processed_annotation = [annotation[0] / original_width,
                                annotation[1] / original_height,
                                annotation[2] / original_width,
                                annotation[3] / original_height]

        processed_images.append(image)
        processed_annotations.append(processed_annotation)

    processed_images = np.array(processed_images)
    processed_annotations = np.array(processed_annotations)

    return processed_images, processed_annotations

def create_model(desired_height, desired_width):
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(desired_height, desired_width, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  # Output layer for temperature prediction
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size, epochs):
    try:
        # Convert data to appropriate types
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Train the model
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
                  callbacks=[EpochLossLogger()])

    except Exception as e:
        print("Error:", e)


def evaluate_model(model, X_test, y_test):
    # Convert temperature array to NumPy array
    y_test = np.array(y_test)

    test_loss = model.evaluate(X_test, y_test)
    print('Test Loss:', test_loss)


def predict_temperature(model, image_path, desired_width, desired_height, original_width, original_height):
    try:
        # Load the new image
        image = cv2.imread(image_path, 0)  # Load the image as grayscale
        if image is None:
            raise ValueError("Failed to load image. Please check the file path.")

        # Resize the new image
        image = cv2.resize(image, (desired_width, desired_height))

        # Normalize pixel values between 0 and 1
        image = image.astype('float32') / 255.0

        # Add an extra dimension for batch
        image = np.expand_dims(image, axis=0)

        # Make the temperature prediction
        predicted_temperature = model.predict(image)
        print('Predicted Temperature:', predicted_temperature)

        # Create a new image with the bounding box
        new_image = cv2.imread(image_path)
        if new_image is None:
            raise ValueError("Failed to load image. Please check the file path.")

        # Determine the bounding box coordinates (adjust as needed)
        x = 10
        y = 10
        width = 50
        height = 50

        # Draw the bounding box on the new image
        cv2.rectangle(new_image, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green bounding box

        # Display the new image with bounding box
        cv2.imshow('New Image', new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except cv2.error as e:
        print("OpenCV Error:", e)

    except Exception as e:
        print("Error:", e)


