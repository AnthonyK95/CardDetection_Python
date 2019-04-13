import numpy as np
import cv2
from matplotlib import pyplot as plt

while True:

    img1 = cv2.imread('./all.png', 0)

    # Initiate ORB Detector => With algorithm and scoreType
    orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)

    # Find the Key Points and Descriptors using ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    output_image = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    cv2.imshow('points', output_image)
    key = cv2.waitKey(0)
    if key == 27:
        break
cv2.destroyAllWindows()