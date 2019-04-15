####### This is the main file

import cv2
import numpy as np
import os
import os.path
import threading


# Create object to read images from camera 0
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 10)
# Initialize ORB object
orb = cv2.ORB_create(nfeatures=100)

imageDir = "images/"
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png"]
valid_image_extensions = [item.lower() for item in valid_image_extensions]

# Comparing the feed
# def ImageClassificationMatcher():
#     for file in os.listdir(imageDir):
#         extension = os.path.splitext(file)[1]
#         if extension.lower() not in valid_image_extensions:
#             continue
#         image_path_list.append(os.path.join(imageDir, file))
#     for imagepath in image_path_list:
#         # Displaying all the images with key Descriptor
#         image = cv2.imread(imagepath, 0)
#         kp1, des1 = orb.detectAndCompute(image, None)
#         if image is not None:
#             keypointsImage = cv2.drawKeypoints(image, kp1, None, color=(0, 255, 0), flags=0)
#             # cv2.imshow(imagepath, keypointsImage)
#             bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
#             # Match descriptors.
#             # matches = bf.match(des1, des2)
#
#
#
#         ## Potential comparizon over here
#
#         elif image is None:
#             print("Error loading" + imagepath)
#             continue
#
# Displaying the data of the images
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
        npArray.append(orb.detectAndCompute(image, None))



while True:
    # Get image from web cam and convert to greyscale

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    computeImageDescriptors()
    # Detect keypoints and descriptors in greyscale image
    keypoints, descriptors = orb.detectAndCompute(img, None)
    # Draw a small red circle with the desired radius
    # at the (x, y) location for each feature found
    for kp in keypoints:
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        cv2.circle(img, (x, y), 2, (0, 0, 255))

    # print(npArray)





        # kp1, des1 = orb.detectAndCompute(image, None)
        # if image is not None:
        #     keypointsImage = cv2.drawKeypoints(image, kp1, None, color=(0, 255, 0), flags=0)



    # Display colour image with detected features
    cv2.imshow("features", img)


    # Sleep infinite loop for ~10ms
    # Exit if user presses <Esc>
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()