from imutils import face_utils
import dlib
import cv2
import numpy

def calcDist(landmarks, image, m, n):
    x = landmarks.part(m).x
    y = landmarks.part(m).y
    e = numpy.array((x, y))
    a = landmarks.part(n).x
    b = landmarks.part(n).y
    o = numpy.array((a, b))
    dist = numpy.linalg.norm(e - o)
    return  dist

p = "shape_predictor_68_face_landmarks.dat"
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
        #for (x, y) in shape:
            #cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        for n in range(0,68):

            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.putText(image, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)





    # Show the image
    cv2.imshow("Output", image)



    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

