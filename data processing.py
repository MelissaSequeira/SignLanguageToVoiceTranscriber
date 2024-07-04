import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

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

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    max_val = np.max(landmarks, axis=0)
    min_val = np.min(landmarks, axis=0)
    normalized_landmarks = (landmarks - min_val) / (max_val - min_val + 1e-6)  # Add small epsilon to avoid division by zero
    return normalized_landmarks.flatten()

print("Video preprocessing complete.")
