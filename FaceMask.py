from imutils import face_utils
import dlib
import cv2
import pandas as pd
import openpyxl
import numpy
import math
import csv

# let's go code an faces detector(HOG) and after detect the
# landmarks on this detected face


def calcDist(landmarks, image, m, n):
    x = landmarks.part(m).x
    y = landmarks.part(m).y
    e = numpy.array((x, y))
    a = landmarks.part(n).x
    b = landmarks.part(n).y
    o = numpy.array((a, b))
    dist = numpy.linalg.norm(e - o)
    return  dist

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "predictor_/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

l = []

while True:
    # Getting out image by webcam
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        landmarks = predictor(gray, rect)
        shape = face_utils.shape_to_np(landmarks)

        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        left_horizontal=calcDist(landmarks, image, 37, 40)
        right_horizontal=calcDist(landmarks, image, 43, 46)

        distance_1 = calcDist(landmarks, image,38, 42)/left_horizontal
        distance_2 = calcDist(landmarks, image, 39, 41)/left_horizontal
        distance_3 = calcDist(landmarks, image, 48, 44)/right_horizontal
        distance_4 = calcDist(landmarks, image, 45, 47)/right_horizontal

        l.append(round(distance_1,2))
        l.append(round(distance_2,2))
        l.append(round(distance_3,2))
        l.append(round(distance_4,2))


    print (l)

    # Show the image
    image=cv2.putText(image, str(l), (32,414), cv2.FONT_HERSHEY_SIMPLEX ,.5, (0,0,255), 1)
    l = []
    cv2.imshow("Output", image)



    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

