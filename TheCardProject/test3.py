# The working protottpe

import cv2
import numpy as np
import os
import os.path


# Create object to read images from camera 0
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 10)
# Initialize ORB object
# orb = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
orb = cv2.ORB_create(nfeatures=2000)
imageDir = "images/"
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png"]
valid_image_extensions = [item.lower() for item in valid_image_extensions]
npArray = []


def computeImageDescriptors():
    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir, file))
    for imagepath in image_path_list:
        # Displaying all the images with key Descriptor
        image = cv2.imread(imagepath, 0)
        cv2.resize(image,(200,200))
        npArray.append(orb.detectAndCompute(image,None))
        npArray.append(np.array([str(imagepath),orb.detectAndCompute(image, None)]))




if __name__ == '__main__':
    computeImageDescriptors()
    while True:
        re,img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, mask = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)

        keypoints, descriptors = orb.detectAndCompute(mask, None)
        for kp in keypoints:
            x = int(kp.pt[0])
            y = int(kp.pt[1])
            cv2.circle(img, (x, y), 2, (0, 0, 255))
        cv2.imshow("Camera Points", img)
        # for i in range(2):
            # create BFMatcher object
            # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors.
            # matches = bf.match(descriptors, npArray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
