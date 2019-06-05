import cv2
import numpy as np
import sys
import os
import operator
import glob

hbins = 180
sbins = 255
hrange = [0,180]
srange = [0,256]
ranges = hrange+srange

flags=[]

titles = []
all_images_to_compare = []


def loadImage():
    for f in glob.iglob("images\*"):
        flags.append(f)
    print(flags)


list_of_pics=[]
valueCompare=[]
cam = cv2.VideoCapture(0)
# Running once to load the image to the array
loadImage()
while True:



    _, frame = cam.read(1)

    cv2.imshow('Frame',frame)
    list_of_pics=[]
    valueCompare=[]
    k=cv2.waitKey(10)
    if(k==32):
        cv2.imwrite("inputimage.jpg",frame) # TODO the image from the camera
        img = "inputimage.jpg"
        for i in flags:
            base = cv2.imread(img)
            test1 = cv2.imread(i) # TODO the flags to be compared with
            rows,cols = base.shape[:2]
            basehsv = cv2.cvtColor(base,cv2.COLOR_BGR2HSV)
            test1hsv = cv2.cvtColor(test1,cv2.COLOR_BGR2HSV)
            histbase = cv2.calcHist(basehsv,[0,1],None,[180,256],ranges)
            cv2.normalize(histbase,histbase,0,255,cv2.NORM_MINMAX)

            histtest1 = cv2.calcHist(test1hsv,[0,1],None,[180,256],ranges)
            cv2.normalize(histtest1,histtest1,0,255,cv2.NORM_MINMAX)

            comHist=cv2.compareHist(histbase,histtest1,3)
            valueCompare.append(comHist)
            picDict={"comhist":comHist,"name":i}
            list_of_pics.append(picDict)

            newlist = sorted(list_of_pics, key=operator.itemgetter('comhist')) #get the max value of all the compared images
            matched_image=newlist[0]['name']
            print (matched_image)
    elif k == 27:
        break
    cv2.destroyAllWindows()