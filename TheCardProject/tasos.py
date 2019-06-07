# Not working

import cv2
import imutils as imutils
import numpy as np
import glob
import matplotlib.pyplot as plt
# capture frames from a camera
cap = cv2.VideoCapture(0)





def compareImages():
    original = cv2.imread("inputimage1.jpg", 0)
    sift = cv2.xfeatures2d.SIFT_create()
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    Okeypoints ,Odesc= sift.detectAndCompute(original, None)



    all_images_to_compare = []
    titles = []
    for f in glob.iglob("images\*"):
        image = cv2.imread(f)
        titles.append(f)
        all_images_to_compare.append(image)
    for image_to_compare, title in zip(all_images_to_compare, titles):
        kp2,desc2 = sift.detectAndCompute(image_to_compare,None)
        matches = flann.knnMatch(Odesc, desc2, k=2)
        good_points = []
        for m, n in matches:
            if m.distance >= 0.8 * m.distance:
                good_points.append(m)
        number_keypoints = 0
        if len(Okeypoints) <= len(kp2):
            number_keypoints = len(Okeypoints)
            cv2.imshow("Result",cv2.imread(title))
        else:
            number_keypoints = len(kp2)
            cv2.imshow("Dont know", cv2.imread(title))
        print("Title: " + title)
        percentage_similarity = len(good_points) / number_keypoints * 100
        print("Similarity: " + str(int(percentage_similarity)) + "\n")





while True:
    # # reads frames from a camera
    ret, frame = cap.read()
    # # converting BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # # define range of red color in HSV
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([255, 255, 200])
    # # # create a red HSV colour boundary and
    # # # threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # # # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # # # Display an original image
    cv2.imshow('Original', frame)
    # # # finds edges in the input image image and
    # # # marks them in the output map edges
    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('Original', frame)
    #







    # TODO working section
    if cv2.waitKey(2) & 0xFF == ord('s'):
        print("gamw thn mana sou ")
        cv2.imwrite("inputimage%d.jpg" % ret, frame)
        # Starting to compare
        compareImages()




    # Close the window
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


    # Display edges in a frame
    # cv2.imshow('Edges',edges)

    # # Wait for Esc key to stop

# Close the window
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()