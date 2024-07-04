import cv2
import os
import time
import uuid

# Path to store the videos
VIDEOS_PATH = r'C:\Users\Melissa\AppData\Local\Programs\Python\Python312\objdetect\HandVideos'

# Labels for hand gestures
labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
number_of_videos = 4 # Number of videos to record per gesture
video_length = 5  # Length of each video in seconds
fps = 20  # Frames per second

# Create directories if they don't exist
for label in labels:
    os.makedirs(os.path.join(VIDEOS_PATH, label), exist_ok=True)
    
for label in labels:
    cap = cv2.VideoCapture(0)
    print(f'Collecting videos for {label}')
    time.sleep(5)  # Wait before starting to collect videos
    
    for vid_num in range(number_of_videos):
        video_name = os.path.join(VIDEOS_PATH, label, label + '.' + '{}.avi'.format(str(uuid.uuid1())))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_name, fourcc, fps, (640, 480))
        start_time = time.time()
        
        print(f'Collecting video {vid_num+1} for {label}')
        
        while int(time.time() - start_time) < video_length:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                cv2.imshow('frame', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        out.release()
        time.sleep(2)  # Pause between recording each video
        
    cap.release()
    cv2.destroyAllWindows()
