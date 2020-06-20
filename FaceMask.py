import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

frame = cap.read()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("predictor_/shape_predictor_68_face_landmarks.dat")


while True:
    frame=cap.read()
    cv2.imshow("frame", frame)