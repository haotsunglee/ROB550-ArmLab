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

# cv2.imshow("Red3 Colors", rgb_image)
redImage = cv2.inRange(rgb_image, minRed, maxRed)
greenImage = cv2.inRange(rgb_image, minGreen, maxGreen)
blueImage = cv2.inRange(rgb_image, minBlue, maxBlue)
purpleImage = cv2.inRange(rgb_image, minPurple, maxPurple)
orangeImage = cv2.inRange(rgb_image, minOrange, maxOrange)
yellowImage = cv2.inRange(rgb_image, minYellow, maxYellow)

# cv2.imshow("Red Colors", redImage)

kernel = np.ones((4,4),np.uint8)
redImageOpen = cv2.morphologyEx(redImage, cv2.MORPH_OPEN, kernel)
redImageClose = cv2.morphologyEx(redImageOpen, cv2.MORPH_CLOSE, kernel)

# cv2.imshow("Red Colors Close", redImageClose)

_, redImageContours, redImageHierarchy = cv2.findContours(redImageClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(redImageClose, redImageContours, -1, (140,140,140), 2)
# cv2.imshow("Red Contours", redImageClose)


for c in redImageContours:
    moment = cv2.moments(c)

    conX = int(moment["m10"] / moment["m00"])
    conY = int(moment["m01"] / moment["m00"])
    # cv2.circle(redImageClose, (conX, conY), 2, (140, 140, 140), -1)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print(box)
    cv2.drawContours(redImageClose,[box],0,(140,140,140),2)
    cv2.drawContours(rgb_image,[box],0,(35,0,120),2)


cv2.imshow("Red Colors Morph", redImageClose)
cv2.imshow("Final Contours", rgb_image)


while True:
  k = cv2.waitKey(10)
  if k == 27:
    break
cv2.destroyAllWindows()