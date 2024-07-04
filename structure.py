import cv2
import os
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_module = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Import drawing utilities from MediaPipe Hands

directory_path = r'C:\Users\Melissa\AppData\Local\Programs\Python\Python312\objdetect\HandVideos'

# Function to extract hand landmarks using MediaPipe Hands
def extract_hand_landmarks(image):
    results = hands_module.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]  # Return landmarks for the first detected hand
    return None

# Function to draw landmarks and connections on the image
def draw_landmarks(image, landmarks):
    if landmarks:
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

# Function to process videos in a directory and its subdirectories
def process_videos_in_directory(directory_path):
    # Iterate over all directories in the specified path
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            print(f"Processing videos in directory: {subdir}")
            
            # Iterate over all files in the subdirectory
            for filename in os.listdir(subdir_path):
                filepath = os.path.join(subdir_path, filename)
                
                # Check if the file is a video (ends with .avi or .mp4)
                if filename.endswith(".avi") or filename.endswith(".mp4"):
                    cap = cv2.VideoCapture(filepath)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    output_filename = os.path.join(subdir_path, f"processed_{filename}")
                    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Extract hand landmarks
                        landmarks = extract_hand_landmarks(frame)
                        
                        if landmarks:
                            # Draw landmarks and connections on the image
                            draw_landmarks(frame, landmarks)
                        
                        out.write(frame)
                        cv2.imshow("Hand Landmarks", frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows(
                    
                    print(f"Processed {filename} and saved as {output_filename}")
                else:
                    print(f"Ignoring non-video file: {filename}")

# Example usage:
if __name__ == "__main__":
    process_videos_in_directory(directory_path)
