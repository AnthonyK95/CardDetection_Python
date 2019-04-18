import cv2
import numpy as np
import os, os.path

# Folder
imageDir = "images/"
image_path_list = []
cap = cv2.VideoCapture(0)
valid_image_extensions = [".jpg", ".jpeg", ".png"]
valid_image_extensions = [item.lower() for item in valid_image_extensions]
orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
# surf = cv2.xfeatures2d.SURF_create()

camera = cv2.VideoCapture(0)

kp = []
dsc = []
# Drawing the keypoints and returning the descriptors
def something():
    # image_list = []
    # Adding all the images path by spliting it
    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir,file))
    # Looping through the images and computing the descriptors
    for imagepath in image_path_list:
        # Displaying all the images with key Descriptor
        image = cv2.imread(imagepath,0)
        # kp1, des1 = orb.detectAndCompute(image, None)
        # image_list.append(des1)
        sift = cv2.xfeatures2d.SIFT_create()
        k,d = sift.detectAndCompute(image, None)
        kp.append(k)
        dsc.append(d)


        # Looping until the images are over
        if image is not None:
            keypointsImage = cv2.drawKeypoints(image, k, None, color=(0, 255, 0), flags=0)
            # Displaying the keypoints on the image
            cv2.imshow(imagepath, keypointsImage)

        elif image is None:
            print("Error on path loading" + imagepath)

    # return image_list
    
    
           
# Searching live for keypoint matching
# def KeypointFrame(kep,des):
#     # imageArray = something()
#     matchingPoints(des,image_list)

  




def matchingPoints(des11, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des11,des2)
    matches = sorted(matches , key = lambda x:x.distance)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>0:
        print("yolo")
    else:
        print("hello")



if __name__ == "__main__":
    # Running once to find the keypoints
    something()

    while True:
    
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(gray, None)
        keypointsImage = cv2.drawKeypoints(gray, kp1, None, color=(0, 255, 0), flags=0)


        # Sending the keypoints and descriptors for matching
        # KeypointFrame(kp1,des1)
        matchingPoints(des1,dsc)
      

        cv2.imshow("Frame",keypointsImage)    

        # Exiting the Program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # Closing all the windows
    cv2.destroyAllWindows()

