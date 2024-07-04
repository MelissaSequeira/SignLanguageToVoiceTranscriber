import cv2
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib  # To save the LabelEncoder

# Suppress specific warning from protobuf
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Function to extract and normalize hand landmarks
# Function to extract and normalize hand landmarks
def extract_hand_landmarks(image, hands_module):
    results = hands_module.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_single_hand = []
            for lm in hand_landmarks.landmark:
                landmarks_single_hand.extend([lm.x, lm.y, lm.z])
            landmarks.append(normalize_landmarks(landmarks_single_hand))
    return np.array(landmarks) if landmarks else None

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    max_val = np.max(landmarks, axis=0)
    min_val = np.min(landmarks, axis=0)
    normalized_landmarks = (landmarks - min_val) / (max_val - min_val + 1e-6)  # Add small epsilon to avoid division by zero
    return normalized_landmarks.flatten()
def load_images(image_path, labels):
    data, target = [], []
    for label in labels:
        dir_path = os.path.join(image_path, label)
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            continue
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            print(f"Loading image: {img_path}")  # Debug print
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error loading image: {img_path}")
                continue
            landmarks = extract_hand_landmarks(image, hands)
            if landmarks is not None and landmarks.any():
                data.append(landmarks.flatten())  # Flatten the landmarks to make them 1-dimensional
                target.append(label)
    if len(data) == 0 or len(target) == 0:
        raise ValueError("No images loaded. Please check if directories exist and contain images.")
    return np.array(data), np.array(target)

# Specify the image path and labels
IMAGES_PATH = r'C:\Users\Melissa\AppData\Local\Programs\Python\Python312\objdetect\HandFrames'  # Update with your actual path
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Load and preprocess images
try:
    X, y = load_images(IMAGES_PATH, labels)
except ValueError as e:
    print(f"Error loading images: {e}")
    exit(1)

# Convert labels to numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save the LabelEncoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(X[0]),)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Save the model in HDF5 format
model.save('sign_language_model.h5')
