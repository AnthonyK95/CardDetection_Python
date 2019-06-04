import cv2
import numpy as np
import glob

# Capturing Video frames
camera = cv2.VideoCapture(0)
orb = cv2.ORB()
all_images_to_compare = []
titles = []
def findKeyPointsFile():
    for f in glob.glob("images\*"):
        image = cv2.imread(f)
        # titles.append(f)
        # all_images_to_compare.append(image)
        cv2.imshow("d",image)


def theCam():
    _, frame = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    retval, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)


    # Todo: Match the templates


    cv2.imshow("yolo",thresh)


# while True:
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
# # Releasing Camera and Closing windows
# camera.release()
# cv2.destroyAllWindows()


findKeyPointsFile()