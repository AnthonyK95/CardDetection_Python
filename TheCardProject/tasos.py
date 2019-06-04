import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
# capture frames from a camera
cap = cv2.VideoCapture(1)




def compareImages():
    inputImage = cv2.imread("inputimage1.jpg",0)
    orb= cv2.xfeatures2d.SURF_create()

    # orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
    kInput1, dInput1 = orb.detectAndCompute(inputImage, None)
    output_image = cv2.drawKeypoints(inputImage, kInput1, None, color=(0, 255, 0), flags=0)
    all_images_to_compare = []
    titles = []
    for f in glob.iglob("images\*"):
        image = cv2.imread(f)
        titles.append(f)
        all_images_to_compare.append(image)
    for image_to_compare, title in zip(all_images_to_compare, titles):

        hsv = cv2.cvtColor(image_to_compare, cv2.COLOR_BGR2HSV)
        lower_red = np.array([30, 160, 50])
        upper_red = np.array([255, 255, 180])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(image_to_compare,image_to_compare, mask=mask)
        edges = cv2.Canny(image_to_compare, 100, 200)

        kInput12, dInput12 = orb.detectAndCompute(edges, None)
        # images = cv2.drawKeypoints(image_to_compare, kInput12, None, color=(0, 255, 0), flags=0)
        # cv2.imshow(title,images)

        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(dInput1, dInput12)
        flann = cv2.FlannBasedMatcher(kInput1, kInput12)
        matches = flann.knnMatch(dInput1, dInput12, k=2)

        # The matches with shorter distance are the ones we want.
        matches = sorted(matches, key=lambda x: x.distance)


        result = cv2.drawMatches(inputImage,kInput1,image_to_compare,kInput12,matches[:10], None,flags=2)

        # Display the best matching points
        plt.rcParams['figure.figsize'] = [14.0, 7.0]
        plt.title('Best Matching Points')
        plt.imshow(result)
        plt.show()
    # Displaying Only
    cv2.imshow("Input",output_image)















# loop runs if capturing has been initialized
while True:

    # reads frames from a camera
    ret, frame = cap.read()

    # converting BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([30, 100, 50])
    upper_red = np.array([255, 255, 180])

    # create a red HSV colour boundary and
    # threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Display an original image
    cv2.imshow('Original', frame)

    # finds edges in the input image image and
    # marks them in the output map edges
    edges = cv2.Canny(frame, 100, 200)

    if cv2.waitKey(2) & 0xFF == ord('s'):
        print("gamw thn mana sou ")
        cv2.imwrite("inputimage%d.jpg" % ret, edges)
        # Starting to compare
        compareImages()




    # Close the window
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


    # Display edges in a frame
    cv2.imshow('Edges',edges)

    # # Wait for Esc key to stop
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Close the window
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()