from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# Defined Variables
camera = cv2.VideoCapture(0)




# Capture image frame
def captureImage(captured_frame):
    cv2.imwrite("image.jpg",captured_frame)
    print("Writing image to file")




def cropImage():
    image = cv2.imread("image.jpg")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)

    # Perform the image
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





# Releasing the camera
camera.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()
