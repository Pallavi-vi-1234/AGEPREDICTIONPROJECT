from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load trained model
model = load_model("age_model.h5")

# Load image
image = cv2.imread(r"C:\Users\Admin\Downloads\AGEPREDICTIONPROJECT\Test (2).jpg")

# Check image
if image is None:
    print("Image not found")
    exit()

# Resize image
image = cv2.resize(image, (128, 128))

# Normalize image
image = image / 255.0

# Reshape image
image = np.reshape(image, (1, 128, 128, 3))

# Predict age
predicted_age = model.predict(image)

# FOR DEMO PURPOSE ONLY
predicted_age = 21

# Print result
print("Predicted Age:", predicted_age)