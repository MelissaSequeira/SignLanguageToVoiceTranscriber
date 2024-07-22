import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\Melissa\AppData\Local\Programs\Python\Python311\gitproj\SignLanguageToVoiceTranscriber\keras_model.h5", 
                        r"C:\Users\Melissa\AppData\Local\Programs\Python\Python311\gitproj\SignLanguageToVoiceTranscriber\labels.txt")
offset = 20
imgSize = 300
counter = 0
text_speech = pyttsx3.init()

labels = ["Hello", "Yes", "Thank you", "I Love You", "Awake", "Again", "Baby", "Bad", "Brush", "Call", "Careful", 
          "Discuss", "Front", "Good", "Guilty", "Like", "Love", "No", "Peace", "Pen", "See you Later"]

answer = ""  # Initialize answer variable

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

        answer = labels[index]  # Update the answer with the detected label

    text_speech.say(answer)
    text_speech.runAndWait()

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
