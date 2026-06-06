import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------------
# Dataset folder settings
# -----------------------------------
# The dataset folder should contain subfolders named with age labels:
# dataset/18, dataset/19, dataset/20, etc.

dataset_path = "dataset"
image_width = 128
image_height = 128

# Lists to store image data and labels
images = []
ages = []

# -----------------------------------
# Load all images from dataset folders
# -----------------------------------
if not os.path.isdir(dataset_path):
    print("Dataset folder 'dataset' not found.")
    print("Create a folder named 'dataset' in the project root.")
    print("Inside it, add age-labelled folders like:")
    print("  dataset/18/")
    print("  dataset/19/")
    print("Then put face images inside each age folder.")
    sys.exit(1)

for age_folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, age_folder)

    if not os.path.isdir(folder_path):
        continue

    # Skip folders that are not integer age labels
    try:
        age_label = int(age_folder)
    except ValueError:
        print(f"Skipping non-age folder: {age_folder}")
        continue

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            continue

        # Read image using OpenCV
        image = cv2.imread(file_path)

        # Handle invalid or unreadable images safely
        if image is None:
            print(f"Skipping invalid image: {file_path}")
            continue

        try:
            # Resize image to 128x128
            image = cv2.resize(image, (image_width, image_height))
        except Exception as error:
            print(f"Failed to resize image {file_path}: {error}")
            continue

        # Normalize image pixel values to [0, 1]
        image = image.astype("float32") / 255.0

        images.append(image)
        ages.append(age_label)

# Convert lists to NumPy arrays
X = np.array(images, dtype="float32")
y = np.array(ages, dtype="float32")

print(f"Total Images Loaded: {len(X)}")

if len(X) == 0:
    print("No valid images were loaded.")
    print("Make sure the dataset folder has subfolders named by age labels and valid image files.")
    sys.exit(1)

# -----------------------------------
# Split dataset into train and test sets
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"Training images: {len(X_train)}")
print(f"Test images: {len(X_test)}")

# -----------------------------------
# Build CNN model for age prediction
# -----------------------------------
model = Sequential()

# First convolutional block
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(image_height, image_width, 3),
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional block
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional block
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten convolutional output to feed into Dense layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation="relu"))

# Output layer predicts a single age value
model.add(Dense(1, activation="linear"))

# -----------------------------------
# Compile the model
# -----------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mae"],
)

# -----------------------------------
# Train the model for 10 epochs
# -----------------------------------
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
)

# -----------------------------------
# Save the trained model
# -----------------------------------
model.save("age_prediction_model.keras")
print("Model saved as age_prediction_model.keras")

# -----------------------------------
# Predict ages on the test dataset
# -----------------------------------
predictions = model.predict(X_test).flatten()

# -----------------------------------
# Custom accuracy: correct if prediction within ±5 years
# -----------------------------------
correct_count = 0
for predicted_age, true_age in zip(predictions, y_test):
    if abs(predicted_age - true_age) <= 5:
        correct_count += 1

accuracy = (correct_count / len(y_test)) * 100
print(f"Accuracy: {accuracy:.2f}%")
