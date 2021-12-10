#!/usr/bin/python
# checkout https://learnopencv.com/color-spaces-in-opencv-cpp-python/
import argparse
import sys
import cv2
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-d", "--depth", required = True, help = "Path to the depth image")
args = vars(ap.parse_args())
rgb_image = cv2.imread(args["image"])
depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

#Segment the image based on color or depth
#Let's define high and lows for all the colors
minRed = np.array([15, 10, 75])
maxRed = np.array([35, 30, 120])

minGreen = np.array([15, 55, 0])
maxGreen = np.array([55, 90, 20])

minBlue = np.array([75, 40, 0])
maxBlue = np.array([100, 60, 25])

minPurple = np.array([70, 45, 60])
maxPurple = np.array([110, 70, 100])

minOrange = np.array([25, 65, 140])
maxOrange = np.array([55, 100, 200])

minYellow = np.array([0, 145, 190])
maxYellow = np.array([40, 205, 245])

redImage = cv2.inRange(rgb_image, minRed, maxRed)
greenImage = cv2.inRange(rgb_image, minGreen, maxGreen)
blueImage = cv2.inRange(rgb_image, minBlue, maxBlue)
purpleImage = cv2.inRange(rgb_image, minPurple, maxPurple)
orangeImage = cv2.inRange(rgb_image, minOrange, maxOrange)
yellowImage = cv2.inRange(rgb_image, minYellow, maxYellow)

#cv2.imshow("Red Colors", redImage)
#cv2.imshow("Green Colors", greenImage)
#cv2.imshow("Blue Colors", blueImage)
#cv2.imshow("Purple Colors", purpleImage)
#cv2.imshow("Orange Colors", orangeImage)
#cv2.imshow("Yellow Colors", yellowImage)

#remove noise from segmented images
kernel = np.ones((4,4),np.uint8)
redImageOpen = cv2.morphologyEx(redImage, cv2.MORPH_OPEN, kernel)
redImageClose = cv2.morphologyEx(redImageOpen, cv2.MORPH_CLOSE, kernel)
greenImageOpen = cv2.morphologyEx(greenImage, cv2.MORPH_OPEN, kernel)
greenImageClose = cv2.morphologyEx(greenImageOpen, cv2.MORPH_CLOSE, kernel)
blueImageOpen = cv2.morphologyEx(blueImage, cv2.MORPH_OPEN, kernel)
blueImageClose = cv2.morphologyEx(blueImageOpen, cv2.MORPH_CLOSE, kernel)
purpleImageOpen = cv2.morphologyEx(purpleImage, cv2.MORPH_OPEN, kernel)
purpleImageClose = cv2.morphologyEx(purpleImageOpen, cv2.MORPH_CLOSE, kernel)
orangeImageOpen = cv2.morphologyEx(orangeImage, cv2.MORPH_OPEN, kernel)
orangeImageClose = cv2.morphologyEx(orangeImageOpen, cv2.MORPH_CLOSE, kernel)
yellowImageOpen = cv2.morphologyEx(yellowImage, cv2.MORPH_OPEN, kernel)
yellowImageClose = cv2.morphologyEx(yellowImageOpen, cv2.MORPH_CLOSE, kernel)

#cv2.imshow("Red Colors Morph", redImageClose)
#cv2.imshow("Green Colors Morph", greenImageClose)
#cv2.imshow("Blue Colors Morph", blueImageClose)
#cv2.imshow("Purple Colors Morph", purpleImageClose)
#cv2.imshow("Orange Colors Morph", orangeImageClose)
#cv2.imshow("Yellow Colors Morph", yellowImageClose)

#find contours of the segments

_, redImageContours, redImageHierarchy = cv2.findContours(redImageClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
_, greenImageContours, greenImageHierarchy = cv2.findContours(greenImageClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
_, blueImageContours, blueImageHierarchy = cv2.findContours(blueImageClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
_, purpleImageContours, purpleImageHierarchy = cv2.findContours(purpleImageClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
_, orangeImageContours, orangeImageHierarchy = cv2.findContours(orangeImageClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
_, yellowImageContours, yellowImageHierarchy = cv2.findContours(yellowImageClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#cv2.drawContours(redImageClose, redImageContours, -1, (140,140,140), 2)
#cv2.drawContours(greenImageClose, greenImageContours, -1, (140,140,140), 2)
#cv2.drawContours(blueImageClose, blueImageContours, -1, (140,140,140), 2)
#cv2.drawContours(purpleImageClose, purpleImageContours, -1, (140,140,140), 2)
#cv2.drawContours(orangeImageClose, orangeImageContours, -1, (140,140,140), 2)
#cv2.drawContours(yellowImageClose, yellowImageContours, -1, (140,140,140), 2)
#cv2.imshow("Red Colors Morph", redImageClose)
#cv2.imshow("Green Colors Morph", greenImageClose)
#cv2.imshow("Blue Colors Morph", blueImageClose)
#cv2.imshow("Purple Colors Morph", purpleImageClose)
#cv2.imshow("Orange Colors Morph", orangeImageClose)
#cv2.imshow("Yellow Colors Morph", yellowImageClose)
#print(len(redImageContours))
#print(len(greenImageContours))
#print(len(blueImageContours))
#print(len(purpleImageContours))
#print(len(orangeImageContours))
#print(len(yellowImageContours))

#calculate the moments for the contours to find the centroids of
#the blobs in the image

for c in redImageContours:
    moment = cv2.moments(c)

    conX = int(moment["m10"] / moment["m00"])
    conY = int(moment["m01"] / moment["m00"])
    cv2.circle(redImageClose, (conX, conY), 2, (140, 140, 140), -1)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(redImageClose,[box],0,(140,140,140),2)
    cv2.drawContours(rgb_image,[box],0,(35,0,120),2)


for c in greenImageContours:
    moment = cv2.moments(c)

    conX = int(moment["m10"] / moment["m00"])
    conY = int(moment["m01"] / moment["m00"])
    cv2.circle(greenImageClose, (conX, conY), 2, (140, 140, 140), -1)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(greenImageClose,[box],0,(140,140,140),2)
    cv2.drawContours(rgb_image,[box],0,(20,45,0),2)

for c in blueImageContours:
    moment = cv2.moments(c)

    conX = int(moment["m10"] / moment["m00"])
    conY = int(moment["m01"] / moment["m00"])
    cv2.circle(blueImageClose, (conX, conY), 2, (140, 140, 140), -1)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(blueImageClose,[box],0,(140,140,140),2)
    cv2.drawContours(rgb_image,[box],0,(95,20,0),2)

for c in purpleImageContours:
    moment = cv2.moments(c)

    conX = int(moment["m10"] / moment["m00"])
    conY = int(moment["m01"] / moment["m00"])
    cv2.circle(purpleImageClose, (conX, conY), 2, (140, 140, 140), -1)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(purpleImageClose,[box],0,(140,140,140),2)
    cv2.drawContours(rgb_image,[box],0,(105,0,95),2)

for c in orangeImageContours:
    moment = cv2.moments(c)

    conX = int(moment["m10"] / moment["m00"])
    conY = int(moment["m01"] / moment["m00"])
    cv2.circle(orangeImageClose, (conX, conY), 2, (140, 140, 140), -1)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(orangeImageClose,[box],0,(140,140,140),2)
    cv2.drawContours(rgb_image,[box],0,(0,95,195),2)

for c in yellowImageContours:
    moment = cv2.moments(c)

    conX = int(moment["m10"] / moment["m00"])
    conY = int(moment["m01"] / moment["m00"])
    cv2.circle(yellowImageClose, (conX, conY), 2, (140, 140, 140), -1)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(yellowImageClose,[box],0,(140,140,140),2)
    cv2.drawContours(rgb_image,[box],0,(0,200,240),2)

cv2.imshow("Red Colors Morph", redImageClose)
cv2.imshow("Green Colors Morph", greenImageClose)
cv2.imshow("Blue Colors Morph", blueImageClose)
cv2.imshow("Purple Colors Morph", purpleImageClose)
cv2.imshow("Orange Colors Morph", orangeImageClose)
cv2.imshow("Yellow Colors Morph", yellowImageClose)
cv2.imshow("Final Contours", rgb_image)


#use cv2.minAreaRect() to find orientation of block


#use centroid, orientation, depth value, inverse intrinsic matrix
#& inverse extrinsic matrix to find locaitons in workspace

while True:
  k = cv2.waitKey(10)
  if k == 27:
    break
cv2.destroyAllWindows()