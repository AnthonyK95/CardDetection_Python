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

camera = cv2.VideoCapture(0)


# Drawing the keypoints and returning the descriptors
def something():
     # Adding all the images path by spliting it
    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir,file))

    for imagepath in image_path_list:
        # Displaying all the images with key Descriptor
        image = cv2.imread(imagepath,0)
        kp1, des1 = orb.detectAndCompute(image, None)

        # Looping until the images are over
        if image is not None:
            keypointsImage = cv2.drawKeypoints(image, kp1, None, color=(0, 255, 0), flags=0)
            # Displaying the keypoints on the image
            cv2.imshow(imagepath, keypointsImage)

        elif image is None:
            print("Error loading" + imagepath)

    return kp1,des1
           
# Getting the values of the images 
def KeypointMatching():
    print("hello")

if __name__ == "__main__":
    #Running once to find the keypoints
    something()

    while True:
    
        ret,frame = camera.read()
        gray = cv2.cvtColor()


        # Exiting the Program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # Closing all the windows
    cv2.destroyAllWindows()

