import numpy as np
import cv2

# capture object
cap = cv2.VideoCapture(0)

# iterate object
while True:
    # capture frames
    ret, frame = cap.read()

    # display frame result
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# release captured frame
cap.release()
cv2.destroyAllWindows()
