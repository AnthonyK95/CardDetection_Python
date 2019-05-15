import cv2
import numpy as np
import os
import os.path
import _pickle as cPickle
import base64



# Define the folder and global variables
# orb = cv2.ORB_create(nfeatures=2000)
directory = "./images/"
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png"]
valid_image_extensions = [item.lower() for item in valid_image_extensions]
# star = cv2.FeatureDetector_create("STAR")



camera = cv2.VideoCapture(0)

# Mine Functions
ImageDescriptor = []

# Creating the files for the images
def createImageDescriptors():
    for file in os.listdir(directory):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue    
        image_path_list.append(os.path.join(directory,file))

    for imagepath in image_path_list:
        output_image = cv2.imread(imagepath,0)
        
        # Initiate Sift extractor
        sift = cv2.ORB_create(nfeatures=100)

        # Find the keypoints with SIFT
        kp = sift.detect(output_image,None)

        # Compute the descriptors with Sift
        kp, des = sift.compute(output_image, kp)

        # Adding all the descriptors to the project        
        ImageDescriptor.append(des)
    

        
# Getting the keypoints from then camera feed
def genSiftFeaturesCamera(gray_img):
    sift = cv2.ORB_create(nfeatures=100)
    ckp, cdesc = sift.detectAndCompute(gray_img, None)
    return cdesc


if __name__ == "__main__":
    createImageDescriptors()
    print(image_path_list)
    print(ImageDescriptor)
    while True:
        _,frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(genSiftFeaturesCamera(frame), createImageDescriptors())
        matches = sorted(matches, key = lambda x:x.distance)
        
    





        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()    
    cv2.destroyAllWindows()