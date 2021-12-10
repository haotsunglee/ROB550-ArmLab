"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([[930.968269,0,641.5526133,0],[0,927.8618057,358.2265057,0],[0,0,1,0]])
        self.extrinsic_matrix = np.array([[1,0,0,-0.5*25.4],[0,-1,0,7.5*25.4],[0,0,-1,38.35*25.4],[0,0,0,1]])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detectBlocksInDepthImage(self, lower, upper):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        colors = list((
            {'id': 'red', 'color': (110, 25, 30)},
            {'id': 'red', 'color': (122.5, 50, 52.5)},
            {'id': 'orange', 'color': (180, 82.5, 27.5)},
            {'id': 'orange', 'color': (145, 65.5, 25.5)},
            {'id': 'yellow', 'color': (245, 195, 35)},
            {'id': 'yellow', 'color': (200.5, 150, 25)},
            {'id': 'green', 'color': (50, 85, 60)},
            {'id': 'green', 'color': (35, 77.5, 52.5)},
            {'id': 'blue', 'color': (2.5, 47.5, 95)},
            {'id': 'blue', 'color': (17.5, 75, 122.5)},
            {'id': 'violet', 'color': (50, 40, 80)},
            {'id': 'violet', 'color': (50, 37.5, 77.5)})
        )
        def retrieve_area_color(data, contour, labels):
            mask = np.zeros(data.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            # cv2.drawContours(data,[box],0,(200,200,200),2)
            mean = cv2.mean(data, mask=mask)[:3]
            # cv2.putText(self.VideoFrame, str(mean), (conX, conY), font, 0.5, (255,255,255), thickness=1)
            min_dist = (np.inf, None)
            for label in labels:
                d = np.linalg.norm(label["color"] - np.array(mean))
                if d < min_dist[0]:
                    min_dist = (d, label["id"])
            return min_dist[1] 
        
        hsv_image = cv2.cvtColor(self.VideoFrame, cv2.COLOR_RGB2HSV)
        # only big
        # first: 936
        # second: 899
        # third:  860
        # fourth: 823

        # lower = 930 # ground
        # upper = 960
        # lower = 895 # second floor
        # upper = 928
        # lower = 855 # third floor
        # upper = 893
        # lower = 815 # fourth floor
        # upper = 853
        mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        cv2.rectangle(mask, (225,120),(1100,720), 255, cv2.FILLED)
        cv2.rectangle(mask, (590,414),(735,720), 0, cv2.FILLED)
        cv2.rectangle(self.VideoFrame, (225,120),(1100,720), (255, 0, 0), 2)
        cv2.rectangle(self.VideoFrame, (590,414),(735,720), (255, 0, 0), 2)
        thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameRaw, lower, upper), mask)
        kernel = np.ones((4,4),np.uint8)
        threshOpen = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        threshClose = cv2.morphologyEx(threshOpen, cv2.MORPH_CLOSE, kernel)

        _, contours, _ = cv2.findContours(threshClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tempBlockDetection = np.array([])
        self.block_contours = contours
        for c in contours:
            moment = cv2.moments(c)
            conX = int(moment["m10"] / moment["m00"])
            conY = int(moment["m01"] / moment["m00"])
            rect = cv2.minAreaRect(c)

            angle = rect[2]
            
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # color detector based on rgb
            color = retrieve_area_color(self.VideoFrame, c, colors)

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
                {'id' : 'violet', 'color' : 110},
                {'id' : 'violet', 'color' : 123},
                {'id' : 'violet', 'color' : 135}
            ))
            for h in hue_colors:
                d = abs(h["color"] - hue)
                if d < min_dist[0]:
                    min_dist = (d, h["id"])
            hue_color = min_dist[1]


            # cv2.drawContours(threshClose, [box], 0,(140,140,140),2)
            cv2.drawContours(self.VideoFrame,[box],0,(0,200,200),2)
            cv2.putText(self.VideoFrame, hue_color, (conX-30, conY+40), font, 1.0, (0,0,0), thickness=1)
            cv2.putText(self.VideoFrame, str(hue), (conX, conY), font, 0.5, (255,255,255), thickness=1)
            
            x, y = rect[0]
            size = rect[1][0] * rect[1][1] # small : 500~800; large : 1500~1900

            # tranfer to workspace coordinate
            depth = self.DepthFrameRaw[int(y)][int(x)]
            intrinsic = np.array([[930.968269,0,641.5526133],[0,927.8618057,358.2265057],[0,0,1]])
            u = np.array([x, y, 1])
            w_c = np.linalg.inv(intrinsic).dot(u.T)
            w_c = w_c * depth
            w_c = np.array([w_c[0], w_c[1], w_c[2], 1])
            w = np.linalg.inv(self.extrinsic_matrix).dot(w_c)
            # cv2.putText(self.VideoFrame, str(w), (conX+30, conY-40), font, 0.4, (0,0,0), thickness=1)
            # (color, x, y, z, angle, size)
            block  = np.array([color, w[0], w[1], w[2], angle, size])
            tempBlockDetection = np.append(tempBlockDetection, block, axis=0)

        self.block_detections = tempBlockDetection.reshape((tempBlockDetection.shape[0]/6, 6))
        # print(self.block_detections.shape)
        # print(self.block_detections)


class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image

        # self.camera.detectBlocksInDepthImage(930,960)
        # self.camera.processVideoFrame()
        # time.sleep(3)

class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        #for detection in data.detections:
        #    print(detection.id[0])
        #    print(detection.pose.pose.pose.position)
        #    print(detection.pose.pose.pose.orientation)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
