from imutils import face_utils
import playsound
import dlib
import cv2
import numpy


# for calculating distance between points
def calcDist(landmarks, image, m, n):
    x = landmarks.part(m).x
    y = landmarks.part(m).y
    e = numpy.array((x, y))
    a = landmarks.part(n).x
    b = landmarks.part(n).y
    o = numpy.array((a, b))
    dist = numpy.linalg.norm(e - o)
    return dist


# p = our pre-trained dlib model.
p = "predictor_/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

#distance list
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
        # for (x, y) in shape:
        #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # distance between points

        left_eye_horizontal = calcDist(landmarks, image, 36, 39)
        right_eye_horizontal = calcDist(landmarks, image, 42, 45)

        distance_1 = calcDist(landmarks, image, 38, 40) / left_eye_horizontal
        distance_2 = calcDist(landmarks, image, 37, 41) / left_eye_horizontal
        distance_3 = calcDist(landmarks, image, 43, 47) / right_eye_horizontal
        distance_4 = calcDist(landmarks, image, 44, 46) / right_eye_horizontal

        l.append(round(distance_1, 2))
        l.append(round(distance_2, 2))
        l.append(round(distance_3, 2))
        l.append(round(distance_4, 2))

    print (l)

    # Show the image
    image = cv2.putText(image, str(l), (32, 414), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    for i in l:
        t = False
        # if the distance between points is less than threshold
        if i < .20:
            print ("Blinked")
            t = True
            playsound.playsound("beep-08b.wav")
            image = cv2.putText(image, "Blinked", (418, 414), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
            break
        if t:
            break
    l = []
    cv2.imshow("Camera", image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
