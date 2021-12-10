
#!/usr/bin/python
""" Example: 

python block_detector_test.py -i image_blocks.png -d depth_blocks.png -l 945 -u 975

"""
import argparse
import sys
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
colors = list((
    {'id': 'red', 'color': (30, 25, 110)},
    {'id': 'red', 'color': (52.5, 50, 122.5)},
    {'id': 'orange', 'color': (27.5, 82.5, 180)},
    {'id': 'orange', 'color': (17.5, 65.5, 155)},
    {'id': 'yellow', 'color': (35, 195, 245)},
    {'id': 'yellow', 'color': (20, 180, 227.5)},
    {'id': 'green', 'color': (60, 85, 50)},
    {'id': 'green', 'color': (52.5, 77.5, 35)},
    {'id': 'blue', 'color': (95, 47.5, 2.5)},
    {'id': 'blue', 'color': (122.5, 75, 17.5)},
    {'id': 'violet', 'color': (80, 40, 50)},
    {'id': 'violet', 'color': (77.5, 37.5, 50)})
)

def retrieve_area_color(data, contour, labels):
    mask = np.zeros(data.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # cv2.drawContours(data,[box],0,(200,200,200),2)
    mean = cv2.mean(data, mask=mask)[:3]
    min_dist = (np.inf, None)
    for label in labels:
        d = np.linalg.norm(label["color"] - np.array(mean))
        if d < min_dist[0]:
            min_dist = (d, label["id"])
    return min_dist[1] 


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-d", "--depth", required = True, help = "Path to the depth image")
ap.add_argument("-l", "--lower", required = True, help = "lower depth value for threshold")
ap.add_argument("-u", "--upper", required = True, help = "upper depth value for threshold")
args = vars(ap.parse_args())
lower = int(args["lower"])
upper = int(args["upper"])
rgb_image = cv2.imread(args["image"])
hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Threshold window", cv2.WINDOW_NORMAL)
"""mask out arm & outside board"""
mask = np.zeros_like(depth_data, dtype=np.uint8)
cv2.rectangle(mask, (220,120),(1100,720), 255, cv2.FILLED)
cv2.rectangle(mask, (590,414),(755,720), 0, cv2.FILLED)
cv2.rectangle(rgb_image, (220,120),(1100,720), (255, 0, 0), 2)
cv2.rectangle(rgb_image, (590,414),(755,720), (255, 0, 0), 2)
thresh = cv2.bitwise_and(cv2.inRange(depth_data, lower, upper), mask)

# cv2.imshow("Threshold window", thresh)

kernel = np.ones((4,4),np.uint8)
threshOpen = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
threshClose = cv2.morphologyEx(threshOpen, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("Threshold Morph", threshClose)


# depending on your version of OpenCV, the following line could be:
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
_, contours, _ = cv2.findContours(threshClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(rgb_image, contours, -1, (0,255,255), 3)
print(len(contours))
for c in contours:
    # print(c)
    moment = cv2.moments(c)
    conX = int(moment["m10"] / moment["m00"])
    conY = int(moment["m01"] / moment["m00"])
    rect = cv2.minAreaRect(c)

    angle = rect[2]
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    color = retrieve_area_color(rgb_image, c, colors)
    
    # color detector based on hue
    hue = hsv_image[int(rect[0][1])][int(rect[0][0])][0]
    min_dist = (np.inf, None)
    hue_colors = list((
    {'id': 'red', 'color': 0},
    {'id' : 'red', 'color' : 180},
    {'id' : 'orange', 'color' : 10},
    {'id' : 'yellow', 'color' : 23},
    {'id' : 'green', 'color' : 70},
    {'id' : 'blue', 'color' : 105},
    {'id' : 'violet', 'color' : 128}
    ))
    for h in hue_colors:
        d = abs(h["color"] - hue)
        if d < min_dist[0]:
            min_dist = (d, h["id"])
    hue_color = min_dist[1] 


    cv2.drawContours(threshClose, [box], 0,(140,140,140),2)
    cv2.drawContours(rgb_image,[box],0,(0,200,200),2)
    cv2.putText(rgb_image, hue_color, (conX-30, conY+40), font, 1.0, (0,0,0), thickness=1)
    cv2.putText(rgb_image, str(depth_data[int(rect[0][1])][int(rect[0][0])]), (conX+30, conY-40), font, 0.4, (0,0,0), thickness=1)
    cv2.putText(rgb_image, str(int(angle)), (conX, conY), font, 0.5, (255,255,255), thickness=1)
    # print(depth_data[int(rect[0][0])][int(rect[0][1])])
cv2.imshow("Threshold window", threshClose)
cv2.imshow("Image window", rgb_image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
