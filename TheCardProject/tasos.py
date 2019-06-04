import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
# capture frames from a camera
cap = cv2.VideoCapture(0)



#
# def compareImages():
#     inputImage = cv2.imread("inputimage1.jpg",0)
#     # orb= cv2.xfeatures2d.SURF_create()
#
#     orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
#     kInput1, dInput1 = orb.detectAndCompute(inputImage, None)
#     output_image = cv2.drawKeypoints(inputImage, kInput1, None, color=(0, 255, 0), flags=0)
#     all_images_to_compare = []
#     titles = []
#     for f in glob.iglob("images\*"):
#         image = cv2.imread(f)
#         titles.append(f)
#         all_images_to_compare.append(image)
#     for image_to_compare, title in zip(all_images_to_compare, titles):
#
#         hsv = cv2.cvtColor(image_to_compare, cv2.COLOR_BGR2HSV)
#         lower_red = np.array([30, 160, 50])
#         upper_red = np.array([255, 255, 180])
#         mask = cv2.inRange(hsv, lower_red, upper_red)
#         res = cv2.bitwise_and(image_to_compare,image_to_compare, mask=mask)
#         edges = cv2.Canny(image_to_compare, 100, 200)
#
#         kInput12, dInput12 = orb.detectAndCompute(edges, None)
#
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
#         matches = bf.match(dInput1, dInput12)
#
#         # The matches with shorter distance are the ones we want.
#         matches = sorted(matches, key=lambda x: x.distance)
#
#
#         result = cv2.drawMatches(inputImage,kInput1,image_to_compare,kInput12,matches[:10], None,flags=2)
#
#         # Display the best matching points
#         plt.rcParams['figure.figsize'] = [14.0, 7.0]
#         plt.title('Best Matching Points')
#         plt.imshow(result)
#         plt.show()
#     # Displaying Only
#     cv2.imshow("Input",output_image)
#
#













# loop runs if capturing has been initialized




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
            if m.distance >= 0.8 * n.distance:
                good_points.append(m)
        number_keypoints = 0
        if len(Okeypoints) >= len(kp2):
            number_keypoints = len(Okeypoints)
            cv2.imshow("Result",cv2.imread(title))
        else:
            number_keypoints = len(kp2)
        print("Title: " + title)
        percentage_similarity = len(good_points) / number_keypoints * 100
        print("Similarity: " + str(int(percentage_similarity)) + "\n")



while True:
    # reads frames from a camera
    ret, frame = cap.read()
    # converting BGR to HSV
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # define range of red color in HSV
    # lower_red = np.array([30, 100, 50])
    # upper_red = np.array([255, 255, 180])
    # # create a red HSV colour boundary and
    # # threshold HSV image
    # mask = cv2.inRange(hsv, lower_red, upper_red)
    # # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    # # Display an original image
    # cv2.imshow('Original', frame)
    # # finds edges in the input image image and
    # # marks them in the output map edges
    # edges = cv2.Canny(frame, 100, 200)


    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original', frame)










    if cv2.waitKey(2) & 0xFF == ord('s'):
        print("gamw thn mana sou ")
        cv2.imwrite("inputimage%d.jpg" % ret, img_gray)
        # Starting to compare
        compareImages()




    # Close the window
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


    # Display edges in a frame
    # cv2.imshow('Edges',edges)

    # # Wait for Esc key to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the window
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()