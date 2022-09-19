import numpy as np
import cv2

face_cascades = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
profile_cascades = cv2.CascadeClassifier('data/haarcascade_profileface.xml')

cap = cv2.VideoCapture(0)

color = (255, 0, 0)
stroke = 2
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    profiles = profile_cascades.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        print(x, y, w, h)
        roi_g = gray[y:y+h, x:x+w]
        cv2.imwrite('face.png', roi_g)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
