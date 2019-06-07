from imutils import perspective
from imutils import contours
import glob
import numpy as np
import imutils
import cv2

# Defined Variables
camera = cv2.VideoCapture(0)


# Capture image frame and Save it
def captureImage(captured_frame):
    cv2.imwrite("image.jpg",captured_frame)
    print("Writing image to file")



# Cropping image frame and save it
def cropImage():
    image = cv2.imread("image.jpg")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)

    # Perform the image filtering
    edged = cv2.Canny(gray,50,100)
    edged = cv2.dilate(edged,None,iterations=1)
    edged = cv2.erode(edged,None,iterations=1)

    # Find contours in the edged map
    cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < 2000:
            continue
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        orig = image.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 5)
        for (xA,yA) in list(box):
            cv2.circle(orig, (int(xA), int(yA)), 9, (0,0,255), -1)
            cv2.imshow("Image", cv2.resize(orig,(800,600)))
            x, y, w, h = cv2.boundingRect(c)
            roi = image[y:y + h, x:x + w]
            cv2.imwrite("kurwa.jpg", roi, (800, 600))




# Comparing Images
def compareImages():
    titles = []
    template = []

    sift = cv2.xfeatures2d.SIFT_create()
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Adding filters to the saved image
    saved = cv2.imread("kurwa.jpg")
    grayS = cv2.cvtColor(saved,cv2.COLOR_BGR2GRAY)
    graySS = cv2.GaussianBlur(grayS,(7,7),0)
    Onek ,OneD = sift.detectAndCompute(graySS, None)


    for f in glob.iglob("images\*"):
        image = cv2.imread(f)
        titles.append(f)
        template.append(image)
    for template, title in zip(template, titles):

        convertedTemplate = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        convertedTemplate = cv2.GaussianBlur(template,(7,7),0)
        twoK, twoD = sift.detectAndCompute(convertedTemplate, None)

        matches = flann.knnMatch(OneD, twoD, k=2)
        good_points = []
        for m, n in matches:
            if m.distance <= 0.5 * m.distance:
                good_points.append(m)
        number_keypoints = 0
        if len(Onek) <= len(twoK):
            number_keypoints = len(Onek)
            cv2.imshow("Result", cv2.imread(title))
        else:
            number_keypoints = len(twoK)
            cv2.imshow("Dont know", cv2.imread(title))







while True:


    # Showing the frame
    ret, frame = camera.read()
    cv2.imshow("Frame",frame)
    # Working switch case to close
    key = cv2.waitKey(2)
    if key == 27:
        break
    elif key == 13:
        contouredArea = frame
        # Sending the image to save
        captureImage(contouredArea)
        # Cropping Image
        cropImage()
        compareImages()





# Releasing the camera
camera.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()
