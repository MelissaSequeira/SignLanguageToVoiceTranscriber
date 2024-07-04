import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress specific warning from protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    max_val = np.max(landmarks, axis=0)
    min_val = np.min(landmarks, axis=0)
    normalized_landmarks = (landmarks - min_val) / (max_val - min_val)
    return normalized_landmarks.flatten()

# Function to extract and normalize hand landmarks using MediaPipe Hands
def extract_hand_landmarks(image, hands_module):
    results = hands_module.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return normalize_landmarks(landmarks)
    return None

# Load the pre-trained model
model = load_model(r'C:\Users\Melissa\AppData\Local\Programs\Python\Python312\objdetect\sign_language_model.h5')

# Function to predict gesture
def predict_gesture(model, landmarks, label_encoder):
    if landmarks is not None and landmarks.size > 0:
        input_data = np.expand_dims(landmarks, axis=0)
        predictions = model.predict(input_data)
        predicted_label_idx = np.argmax(predictions)
        predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
        return predicted_label
    return None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Video capture setup
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or another index for other cameras

# Labels for gestures
labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Label encoder to convert labels to numerical format
label_encoder = LabelEncoder()
label_encoder.fit(labels)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Failed to read frame.")
        break
    
    # Process the frame to detect hand landmarks
    landmarks = extract_hand_landmarks(frame, hands)
    
    if landmarks is not None:
        # Predict gesture
        predicted_label = predict_gesture(model, landmarks, label_encoder)
        
        if predicted_label:
            # Draw landmarks on the frame
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
            # Display predicted gesture label
            gesture_text = f"Predicted Gesture: {predicted_label}"
            cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Sign Language Recognition', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

