"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
from camera import Camera
from kinematics import FK_dh, FK_pox, get_pose_from_T, IK_geometric
import rospy
import cv2
from config_parse import *
from rospy.timer import sleep
import math

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]
        self.teachopengrip = False
        self.teachclosegrip = False
        self.pickandplacestatemachine = 0 #0 if the block is not picked, 1 if robot is currently holding the block
        self.yoffset = 25
        self.xoffset = -35

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == "teach":
            self.teach(False, False)
        
        if self.next_state == "stopteach":
            self.stopteach()

        if self.next_state == "repeat":
            self.repeat()
        
        if self.next_state == "teachclosegrip":
            self.teach(False, True)

        if self.next_state == "teachopengrip":
            self.teach(True, False)

        if self.next_state == "pickandplace":
            self.pickandplace()

        if self.next_state == "comp1":
            self.comp1()

        if self.next_state == "comp2":
            self.comp2()

        if self.next_state == "comp3":
            self.comp3()

        if self.next_state == "comp4":
            self.comp4()

        if self.next_state == "comp5":
            self.comp5()




    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"

        for x in self.waypoints:
            if(self.next_state == "estop"):
                break
            self.rxarm.set_positions(x)
            rospy.sleep(5)

        if(self.next_state == "estop"):
            print("Going to E Stop")
        else:
            self.next_state = "idle"
        

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"
        """TODO Perform camera calibration routine here"""
        data = self.camera.tag_detections
        april_tags = np.zeros((4,3))

        # for detection in data.detections:
        #    print(detection.id[0])
        #    print(detection.pose.pose.pose.position)
        # #    print(detection.pose.pose.pose.orientation)
        for detection in data.detections:
           april_tags[detection.id[0]-1][0] = detection.pose.pose.pose.position.x/detection.pose.pose.pose.position.z
           april_tags[detection.id[0]-1][1] = detection.pose.pose.pose.position.y/detection.pose.pose.pose.position.z
           april_tags[detection.id[0]-1][2] = 1

        intrinsic = np.array([[930.968269,0,641.5526133],[0,927.8618057,358.2265057],[0,0,1]])
        image_points = np.dot(intrinsic, april_tags.T)
        image_points = image_points[:2].T
        image_points = np.flip(image_points, 0)
        #print(image_points)
        world_points = np.array([(-5,5.5,0),(5,5.5,0),(5,-0.5,0),(-5,-0.5,0)])*50
        dist_coeffs = np.array([0.124025, -0.183703, -0.00015533333, -0.003647, 0])
        success, rotation_vect, translation_vect = cv2.solvePnP(world_points, image_points, intrinsic, dist_coeffs, flags=0)
        rotation_matrix = cv2.Rodrigues(rotation_vect)[0]
        #print(rotation_matrix)
        #print(translation_vect)
        new_extrinsic = np.column_stack((rotation_matrix, translation_vect))
        new_extrinsic = np.row_stack((new_extrinsic, [0,0,0,1]))
        print(new_extrinsic)
        self.camera.extrinsic_matrix = new_extrinsic

        
        self.status_message = "Calibration - Completed Calibration"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        self.current_state = "detect"
        self.next_state = "idle"
        self.camera.detectBlocksInDepthImage(930,960)
        rospy.sleep(1)

    def teach(self, openGrip, closeGrip):
        """!
        @brief      Gets the user input to record a path
        """
        
        self.rxarm.disable_torque()
        print("Teach: Disabled Torque")
        self.current_state = "teach"
        self.next_state = "teach"
        self.status_message = "Currently recording, press stop teach to finish recording"
        f = open("recordedPath.txt", "a")
        #Idea, record the positions as often as possible into a text file, and then also
        # add Close or Open whenever the gripper moves. This should give us the data we need.

        #sometimes you have too many characters, so need to remove the new line
        currentPos = str(self.rxarm.get_positions())
        f.write(currentPos.replace("\n",""))
        print(currentPos.replace("\n",""))
        f.write("\n")
  
        #writes close and open to file based on button presses
        if(closeGrip == True):
            self.rxarm.enable_torque()
            self.rxarm.close_gripper()
            self.rxarm.disable_torque()
            print("Close\n")
            f.write("Close\n")

        if(openGrip == True):
            self.rxarm.enable_torque()
            self.rxarm.open_gripper()
            self.rxarm.disable_torque()
            print("Open\n")
            f.write("Open\n")

        rospy.sleep(0.1)
                
        f.close()

    def stopteach(self):
        """!
        @brief      Stops the learning
        """
        self.current_state = "stopteach"
        self.next_state = "idle"
        self.rxarm.enable_torque()
        self.status_message = "Teaching Stopped"

    def repeat(self):
        """!
        @brief      Repeat the path recorded in teach.
        """
        self.current_state = "repeat"
        self.next_state = "idle"
        self.rxarm.set_moving_time(0.1)
        self.rxarm.set_accel_time(.01)

        """TODO Perform playback here"""
        f = open("recordedPath.txt", "r")
        currentLine = f.readline()
        while(len(currentLine) != 0):
            print(currentLine)
            if(self.next_state == "estop"):
                    break
            #Check for open and close commands
            if(currentLine == "Close\n"):
                self.rxarm.close_gripper()
            elif(currentLine == "Open\n"):
                self.rxarm.open_gripper()
            else:
                #remove ends of line to get just the data
                currentLine2 = currentLine.replace("[", "")
                currentLine3 = currentLine2.replace("]","")
                #split up line and set it
                [joint1, joint2, joint3, joint4, joint5] = currentLine3.split()
                currentWaypoint = [float(joint1), float(joint2), float(joint3), float(joint4), float(joint5)]
                self.rxarm.set_positions(currentWaypoint)

            currentLine = f.readline()

        print("Finished Playback")
        self.status_message = "playback of teaching finished"

    def clamp(self,angle):

        while angle > np.pi/2:
            angle -= 1 * np.pi
        while angle <= -np.pi/2:
            angle += 1 * np.pi
        return angle

    def pickandplace(self):
        """!
        @brief      Go to point from click and pick up block. Drop at next placement.
                    This will automatically go to a place that was clicked on the screen if there has been any clicking before this, be warned
                    It will also stop running once the block is placed
        """
        #open gripper and setup movement
        self.current_state = "pickandplace"
        self.next_state = "pickandplace"
        self.rxarm.set_moving_time(3)
        self.rxarm.set_accel_time(.5)

        intrinsic = np.array([[930.968269,0,641.5526133],[0,927.8618057,358.2265057],[0,0,1]])
        #extrinsic = np.array([[1,0,0,50.8],[0,-1,0,190.5],[0,0,-1,971.55],[0,0,0,1]])
        extrinsic = self.camera.extrinsic_matrix
        #print(extrinsic)
        u = np.array([self.camera.last_click[0], self.camera.last_click[1], 1])
        z = self.camera.DepthFrameRaw[self.camera.last_click[1]][self.camera.last_click[0]]

        w_c = np.linalg.inv(intrinsic).dot(u.T)
        w_c = w_c * z
        w_c = np.array([w_c[0], w_c[1], w_c[2], 1])
        w = np.linalg.inv(extrinsic).dot(w_c)

        # find the angle of the block you click
        blocks_detections = self.camera.block_detections
        print(blocks_detections.shape)
        min_dist = (np.inf, None)
        index = 0
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3]), 1])
            d = np.linalg.norm(np.subtract(block_position,w))
            if d < min_dist[0]:
                min_dist = (d, index)
            index = index + 1
        index = min_dist[1]
        angle = float(blocks_detections[index][4])


        # print("angle by detector:", angle)
        print("new click:", self.camera.new_click)
        # print("joint name", self.rxarm.joint_names)
        dh_params = parse_dh_param_file("./config/rx200_dh.csv")
        m_mat, s_vector = parse_pox_param_file("./config/rx200_pox.csv")
        #we haven't picked up a block yet
        print('1')
        if(self.pickandplacestatemachine == 0):
            self.rxarm.open_gripper()
            #check if a click has happened and go to click
            if(self.camera.new_click == True):
                self.status_message = "Moving to pick up block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                print('2')
                currentposition = self.rxarm.get_ee_pose()
                print("currentposition: ", currentposition)
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],100,currentposition[3]]))
                if upwaypoint is None:
                    self.camera.new_click = False
                    return
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(5)
                #now go across to pickup location, 100 mm in the air
                newposition = np.array([w[0], w[1], w[2]-10, 0])
                print("newposition:", newposition)
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(5)
                #finally, go down to the ground plane and close the gripper
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
                # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
                # print("base: ", -newwaypoint[0]*180/np.pi)
                # print("angle: ", -angle)
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                
                newwaypoint = np.append(newwaypoint, rotation)
                print("newwaypoint:" ,newwaypoint)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(5)
                self.rxarm.close_gripper()
                #reset camera click variable and update pick and place state machine
                self.camera.new_click = False
                self.pickandplacestatemachine = 1
            else:
                self.status_message = "Click on block to have robot pick it up"

        #we have picked up a block and need to place it
        else:
            #check if a click has happened and go to click
            if(self.camera.new_click == True):
                self.status_message = "Moving to drop off block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                currentposition = self.rxarm.get_ee_pose()
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+100,currentposition[3]]))
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(5)
                #now go across to dropoff location, 100 mm in the air
                newposition = np.array([w[0], w[1], w[2]+10])
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(5)
                #finally, go down to the ground plane and open the gripper
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+10,currentposition[3]]), theta)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(5)
                self.rxarm.open_gripper()
                #reset camera click variable and update pick and place state machine, and go to idle
                self.camera.new_click = False
                self.pickandplacestatemachine = 0
                self.next_state = "idle"
                self.status_message = "pick and place complete, going to idle"
            else:
                self.status_message = "Click on location to have robot drop block off there"

    def comp1(self):
        #Pick n sort
        self.current_state = "comp1"
        self.status_message = "Currently Executing Competition 1"
        self.next_state = "comp1"
        self.rxarm.set_moving_time(2)
        self.rxarm.set_accel_time(.5)
        dh_params = parse_dh_param_file("./config/rx200_dh.csv")

        self.camera.detectBlocksInDepthImage(895,928)
        blocks_detections = self.camera.block_detections
        sleep_time = 2

        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            
            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and close the gripper
            if theta==0:
                newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                newposition[2] = newposition[2]-30 #adjust this offset
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+10,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to dropoff location, depending on size of block.
            #Plan, go to position close, and then go outward and release. Will push blocks farther away from arm if done correctly
            newposition = np.array([0, 0, 0])
            newfarposition = np.array([0, 0, 0])
            if(size == "large"):
                newposition = np.array([150, -124, 35])
                newfarposition = np.array([250, -124, 35])
            else:
                newposition = np.array([-124, -124, 25])
                newfarposition = np.array([-223, -124, 25])
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #go down to the ground plane and open the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2],currentposition[3]]), theta)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            #finally, drag along the ground to farther away from the machine to push earlier block farther away and have a clean spot to place.
            newfarwaypoint, theta = IK_geometric(dh_params,np.array([newfarposition[0]+self.xoffset,newfarposition[1]+self.yoffset,newfarposition[2],currentposition[3]]))
            self.rxarm.set_positions(newfarwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.open_gripper()



        self.camera.detectBlocksInDepthImage(930,960)
        blocks_detections = self.camera.block_detections

        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])

            if block_position[1] <= 10:
                continue

            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            
            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and close the gripper
            if theta==0:
                newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                newposition[2] = newposition[2]-45
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to dropoff location, depending on size of block.
            #Plan, go to position close, and then go outward and release. Will push blocks farther away from arm if done correctly
            newposition = np.array([0, 0, 0])
            newfarposition = np.array([0, 0, 0])
            if(size == "large"):
                newposition = np.array([150, -124, 35])
                newfarposition = np.array([250, -124, 35])
            else:
                newposition = np.array([-124, -124, 25])
                newfarposition = np.array([-223, -124, 25])
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #go down to the ground plane and open the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2],currentposition[3]]), theta)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            #finally, drag along the ground to farther away from the machine to push earlier block farther away and have a clean spot to place.
            newfarwaypoint, theta = IK_geometric(dh_params,np.array([newfarposition[0]+self.xoffset,newfarposition[1]+self.yoffset,newfarposition[2],currentposition[3]]), theta)
            self.rxarm.set_positions(newfarwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.open_gripper()
        self.next_state = "idle"
        self.status_message = "Competition 1 Complete, going to idle"


        #Get the size of block detection array
        #for loop through the array, picking up blocks and putting them in correct spot
        #small to left, large to right
        #in for loop change were block is put down

    def comp2(self):
        #Pick n stack
        self.current_state = "comp2"
        self.status_message = "Currently Executing Competition 2"
        self.next_state = "comp2"
        self.rxarm.set_moving_time(3)
        self.rxarm.set_accel_time(.5)
        
        sleep_time = 3

        #First let's unstack all blocks and put them in a corner for use later
        self.camera.detectBlocksInDepthImage(895,928)
        dh_params = parse_dh_param_file("./config/rx200_dh.csv")
        blocks_detections = self.camera.block_detections
        x_pos = 160
        y_pos = -124
        count = 0
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            
            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and close the gripper
            if theta==0:
                newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                newposition[2] = newposition[2]-30
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+15,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to dropoff location, 100 mm in the air
            newposition = np.array([x_pos, y_pos, 5])
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and open the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+10,currentposition[3]]), theta)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.open_gripper()
            self.next_state = "idle"
            self.status_message = "pick and place complete, going to idle"
            x_pos = x_pos + 80
            count = count + 1
            if count >= 3:
                x_pos = 160
                y_pos = y_pos + 80
                count = 0

        #Go to a safe point before going home
        currentposition = self.rxarm.get_ee_pose()
        print("currentposition: ", currentposition)
        upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],300,currentposition[3]]))
        self.rxarm.set_positions(upwaypoint)
        rospy.sleep(sleep_time)
        #Now we need to find blocks that are far away and place them in a safe place for later.
        home_pos, theta = IK_geometric(dh_params, np.array([0,88,88,0]))
        self.rxarm.set_positions(home_pos)
        rospy.sleep(sleep_time)

        self.camera.detectBlocksInDepthImage(930,960)
        blocks_detections = self.camera.block_detections
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            #Let's see what the IK would return for every block in the field.
            currentposition = self.rxarm.get_ee_pose()
            testpos = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("testpos:", testpos)
            testpoint, theta = IK_geometric(dh_params,np.array([testpos[0]+self.xoffset,testpos[1]+self.yoffset,testpos[2]+100,currentposition[3]]))
            #If theta isn't 0 we're fine, if it is zero we need to do something about that.
            if theta != 0:
                continue

            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and close the gripper
            if theta==0:
                newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                newposition[2] = newposition[2]-30
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to dropoff location, 100 mm in the air
            newposition = np.array([x_pos, y_pos, 5])
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and open the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2],currentposition[3]]), theta)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.open_gripper()
            self.next_state = "idle"
            self.status_message = "pick and place complete, going to idle"
            x_pos = x_pos + 80
            count = count + 1
            if count >= 3:
                x_pos = 160
                y_pos = y_pos + 80
                count = 0
        

        #move all the large blocks into towers
        home_pos, theta = IK_geometric(dh_params, np.array([0,88,88,0]))
        self.rxarm.set_positions(home_pos)
        rospy.sleep(sleep_time)

        self.camera.detectBlocksInDepthImage(930,960)
        blocks_detections = self.camera.block_detections
        pos_index = 0
        x_pos_list = [-150, -215 , -150]
        y_pos_list = [-124, -124, -30]
        height_list = [10, 10, 10]
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            if size == "small":
                continue

            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and close the gripper
            if theta==0:
                newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                newposition[2] = newposition[2]-30
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to dropoff location, 100 mm in the air
            newposition = np.array([x_pos_list[pos_index], y_pos_list[pos_index], height_list[pos_index]])
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and open the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.open_gripper()
            self.next_state = "idle"
            self.status_message = "pick and place complete, going to idle"
            height_list[pos_index] = height_list[pos_index] + 40
            pos_index = pos_index + 1
            if pos_index >= 3:
                pos_index = 0
        
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            if size == "large":
                continue

            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and close the gripper
            if theta==0:
                newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                newposition[2] = newposition[2]-40
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to dropoff location, 100 mm in the air
            newposition = np.array([x_pos_list[pos_index], y_pos_list[pos_index], height_list[pos_index]])
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and open the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.open_gripper()
            self.next_state = "idle"
            self.status_message = "pick and place complete, going to idle"
            height_list[pos_index] = height_list[pos_index] + 30
            pos_index = pos_index + 1
            if pos_index >= 3:
                pos_index = 0


    def comp3(self):
        #line em up
        self.current_state = "comp3"
        self.status_message = "Currently Executing Competition 3"
        self.next_state = "comp3"
        self.rxarm.set_moving_time(3)
        self.rxarm.set_accel_time(.5)

        dh_params = parse_dh_param_file("./config/rx200_dh.csv")

        x_pos = 155
        y_pos = -124
        count = 0
        
        for x in range(3):
            homepos, theta = IK_geometric(dh_params,np.array([0,88,88,0]))
            self.rxarm.set_positions(homepos)
            rospy.sleep(3)

            average = 899+(x*25)
            self.camera.detectBlocksInDepthImage(average-10,average+10)
            blocks_detections = self.camera.block_detections

            for block in blocks_detections:
                block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
                color, angle = block[0], float(block[4]) 
                size = "large" if float(block[5])>1150 else "small"


                if size == "large":
                    continue

                if block_position[1] <= 10:
                    continue
                
                #we haven't picked up a block yet
                self.rxarm.open_gripper()            
                self.status_message = "Moving to pick up block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                currentposition = self.rxarm.get_ee_pose()
                print("currentposition: ", currentposition)
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[1]+300,currentposition[3]]))
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(3)
                #now go across to pickup location, 100 mm in the air
                newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
                print("newposition:", newposition)
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
                if theta==0:
                    newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                    newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                    newposition[2] = newposition[2]-30
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(3)
                #finally, go down to the ground plane and close the gripper
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
                # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
                if(theta != 0):
                    rotation = newwaypoint[0]*180/np.pi+angle
                    rotation = int(rotation) % 90
                    rotation = rotation*np.pi/180
                    newwaypoint = np.append(newwaypoint, rotation)
                # print("newwaypoint:" ,newwaypoint)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(3)
                self.rxarm.close_gripper()

                #we have picked up a block and need to place it
                self.status_message = "Moving to drop off block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                currentposition = self.rxarm.get_ee_pose()
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(3)
                #now go across to dropoff location, depending on size of block.
                #Going to put them in a grid first, then do based on color
                newposition = np.array([x_pos, y_pos, 5])
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(3)
                #go down to the ground plane and open the gripper at the right place
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+10,currentposition[3]]), theta)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(3)
                self.rxarm.open_gripper()
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(3)
                x_pos = x_pos + 90
                count = count + 1
                if count >= 3:
                    x_pos = 155
                    y_pos = y_pos + 90
                    count = 0

        for x in range(2):
            homepos, theta = IK_geometric(dh_params,np.array([0,88,88,0]))
            self.rxarm.set_positions(homepos)
            rospy.sleep(3)

            average = 900+(x*40)
            self.camera.detectBlocksInDepthImage(average-10,average+10)
            blocks_detections = self.camera.block_detections

            for block in blocks_detections:
                block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
                color, angle = block[0], float(block[4]) 
                size = "large" if float(block[5])>1150 else "small"


                if size != "large":
                    continue

                if block_position[1] <= 10:
                    continue
                
                #we haven't picked up a block yet
                self.rxarm.open_gripper()            
                self.status_message = "Moving to pick up block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                currentposition = self.rxarm.get_ee_pose()
                print("currentposition: ", currentposition)
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[1]+300,currentposition[3]]))
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(3)
                #now go across to pickup location, 100 mm in the air
                newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
                print("newposition:", newposition)
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
                if theta==0:
                    newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                    newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                    newposition[2] = newposition[2]-30
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(3)
                #finally, go down to the ground plane and close the gripper
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
                # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
                if(theta != 0):
                    rotation = newwaypoint[0]*180/np.pi+angle
                    rotation = int(rotation) % 90
                    rotation = rotation*np.pi/180
                    newwaypoint = np.append(newwaypoint, rotation)
                # print("newwaypoint:" ,newwaypoint)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(3)
                self.rxarm.close_gripper()

                #we have picked up a block and need to place it
                self.status_message = "Moving to drop off block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                currentposition = self.rxarm.get_ee_pose()
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(3)
                #now go across to dropoff location, depending on size of block.
                #Going to put them in a grid first, then do based on color
                newposition = np.array([x_pos, y_pos, 5])
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(3)
                #go down to the ground plane and open the gripper at the right place
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+10,currentposition[3]]), theta)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(3)
                self.rxarm.open_gripper()
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(3)
                x_pos = x_pos + 90
                count = count + 1
                if count >= 3:
                    x_pos = 155
                    y_pos = y_pos + 90
                    count = 0
        




        home_pos, theta = IK_geometric(dh_params, np.array([0,88,88,0]))
        self.rxarm.set_positions(home_pos)
        rospy.sleep(3)

        #Now that all the blocks are in a grid, let's put them in correct order
        self.camera.detectBlocksInDepthImage(930,960)
        blocks_detections = self.camera.block_detections

        rainbow = {"violet":[],"blue":[],"green":[],"yellow":[],"orange":[],"red":[]}
        sleep_time = 3
        #print(rainbow)
        #after all blocks unstacked and detected again
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            #print(color,' ', rainbow[color])
            rainbow[color] = ([block_position,angle,size])
        #print(rainbow)
        nstack = 0
        b_height = 50
        c_list = ["violet","blue","green","yellow","orange","red"]


        for color in c_list:
            try:
                block = rainbow[color][0]
                angle = rainbow[color][1]
                block_position = block
            except IndexError as error:
                continue
            
            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+200,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(3)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(3)
            #finally, go down to the ground plane and close the gripper
            if theta==0:
                newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                newposition[2] = newposition[2]-40
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(3)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(3)
            #now go across to dropoff location, depending on size of block.
            #Plan, go to position close, and then go outward and release. Will push blocks farther away from arm if done correctly

            if size == "large":
                newposition = np.array([-120, -124, 35])
                newfarposition = np.array([-200, -124, 35])
            else:
                newposition = np.array([-120, 0, 35])
                newfarposition = np.array([-200, 0, 35])

            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(3)
            #go down to the ground plane
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2],currentposition[3]]), theta)
            if size == "large":
                newwaypoint = np.append(newwaypoint, 0.52)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(3)
            #finally, drag along the ground to farther away from the machine to push earlier block farther away and have a clean spot to place.
            newfarwaypoint, theta = IK_geometric(dh_params,np.array([newfarposition[0]+self.xoffset,newfarposition[1]+self.yoffset,newfarposition[2],currentposition[3]]), theta)
            if size == "large":
                newfarwaypoint = np.append(newfarwaypoint,0.52)
            self.rxarm.set_positions(newfarwaypoint)
            rospy.sleep(3)
            self.rxarm.open_gripper()

        self.next_state = "idle"
        self.status_message = "Competition 3 Complete, going to idle"

        #Need to update block detector to be able to scan multiple levels, just do it with big blocks
        #if detector sees blocks at higher levels, pick up and place somewhere else
        #repeat until only blocks on lowest level.
        #If any blocks are in front of arm, move to different spot
        #grab correct blocks and put them in place

    def remove_layer_block(self, low, high,x,y):
        dh_params = parse_dh_param_file("./config/rx200_dh.csv")
        home_pos,theta = IK_geometric(dh_params, np.array([0,88,88,0]))
        self.rxarm.set_positions(home_pos)
        sleep_time = 3
        rospy.sleep(sleep_time)

        self.camera.detectBlocksInDepthImage(low,high)
        blocks_detections = self.camera.block_detections
        pos_index = 0
        pos_list = [x, x+65, x+130]
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            if size == "small":
                continue
            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+50,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]-30,newposition[1]+30,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and close the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]-30,newposition[1]+30,newposition[2]+10,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to dropoff location, 100 mm in the air
            newposition = np.array([pos_list[pos_index], y, 10]) #-124
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]-30,newposition[1]+30,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and open the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]-30,newposition[1]+30,newposition[2]+20,currentposition[3]]), theta)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.open_gripper()
            pos_index += 1

    def comp4(self):
        #stack em high
        self.current_state = "comp4"
        self.status_message = "Currently Executing Competition 4"
        self.next_state = "comp4"
        dh_params = parse_dh_param_file("./config/rx200_dh.csv")
        self.rxarm.set_moving_time(2)
        self.rxarm.set_accel_time(.5)
        sleep_time = 2
        x_pos = 155
        y_pos = -124
        count = 0

        #Unstack all the big blocks first
        for x in range(4):
            homepos, theta = IK_geometric(dh_params,np.array([0,88,88,0]))
            self.rxarm.set_positions(homepos)
            rospy.sleep(sleep_time)

            average = 820+(x*40)
            self.camera.detectBlocksInDepthImage(average-10,average+10)
            blocks_detections = self.camera.block_detections

            for block in blocks_detections:
                block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
                color, angle = block[0], float(block[4]) 
                size = "large" if float(block[5])>1150 else "small"


                if size == "small":
                    continue

                if block_position[1] <= 10:
                    continue
                
                #we haven't picked up a block yet
                self.rxarm.open_gripper()            
                self.status_message = "Moving to pick up block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                currentposition = self.rxarm.get_ee_pose()
                print("currentposition: ", currentposition)
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[1]+300,currentposition[3]]))
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(sleep_time)
                #now go across to pickup location, 100 mm in the air
                newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
                print("newposition:", newposition)
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
                if theta==0:
                    newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                    newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                    newposition[2] = newposition[2]-30
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(sleep_time)
                #finally, go down to the ground plane and close the gripper
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+5,currentposition[3]]), theta)
                # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
                if(theta != 0):
                    rotation = newwaypoint[0]*180/np.pi+angle
                    rotation = int(rotation) % 90
                    rotation = rotation*np.pi/180
                    newwaypoint = np.append(newwaypoint, rotation)
                # print("newwaypoint:" ,newwaypoint)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(sleep_time)
                self.rxarm.close_gripper()

                #we have picked up a block and need to place it
                self.status_message = "Moving to drop off block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                currentposition = self.rxarm.get_ee_pose()
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(sleep_time)
                #now go across to dropoff location, depending on size of block.
                #Going to put them in a grid first, then do based on color
                newposition = np.array([x_pos, y_pos, 5])
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(sleep_time)
                #go down to the ground plane and open the gripper at the right place
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+10,currentposition[3]]), theta)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(sleep_time)
                self.rxarm.open_gripper()
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(sleep_time)
                x_pos = x_pos + 90
                count = count + 1
                if count >= 3:
                    x_pos = 155
                    y_pos = y_pos + 90
                    count = 0
        




        home_pos, theta = IK_geometric(dh_params, np.array([0,88,88,0]))
        self.rxarm.set_positions(home_pos)
        rospy.sleep(sleep_time)    

        #This is the code for once they're unstacked
        self.camera.detectBlocksInDepthImage(930,960)
        dh_params = parse_dh_param_file("./config/rx200_dh.csv")
        blocks_detections = self.camera.block_detections
        height = 10
        rainbow = {"violet":[],"blue":[],"green":[],"yellow":[],"orange":[],"red":[]}
        #print(rainbow)
        #after all blocks unstacked and detected again
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            #print(color,' ', rainbow[color])
            rainbow[color] = ([block_position,angle,size])
        #print(rainbow)
        nstack = 0
        b_height = 50
        c_list = ["violet","blue","green","yellow","orange","red"]
        #print('before color loop')
        x = 0
        y = 0
        for color in c_list:
                try:
                    block = rainbow[color][0]
                    angle = rainbow[color][1]
                    block_position = block
                except IndexError as error:
                    continue
                #we haven't picked up a block yet
                self.rxarm.open_gripper()            
                self.status_message = "Moving to pick up block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                currentposition = self.rxarm.get_ee_pose()
                print("currentposition: ", currentposition)
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(sleep_time)
                #now go across to pickup location, 100 mm in the air
                newposition = np.array([block[0], block[1], block[2]-10, 0])
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]-30,newposition[1]+30,newposition[2]+100,currentposition[3]]))
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(sleep_time)
                #finally, go down to the ground plane and close the gripper
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]-30,newposition[1]+30,newposition[2]+10,currentposition[3]]))
                if(theta != 0):
                    rotation = newwaypoint[0]*180/np.pi+angle
                    rotation = int(rotation) % 90
                    rotation = rotation*np.pi/180
                    newwaypoint = np.append(newwaypoint, rotation)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(sleep_time)
                self.rxarm.close_gripper()

                #we have picked up a block and need to place it
                self.status_message = "Moving to drop off block"
                #IK stuff to make the robot go to the location. go up, across, then down
                #get current position of end effector and go up to 100 mm
                currentposition = self.rxarm.get_ee_pose()
                upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
                self.rxarm.set_positions(upwaypoint)
                rospy.sleep(sleep_time)
                #now go across to dropoff location, 100 mm in the air
                newposition = np.array([-150-x, -124-y, nstack*b_height+20])
                newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]-30,newposition[1]+30,newposition[2]+100,currentposition[3]]))
                self.rxarm.set_positions(newupwaypoint)
                rospy.sleep(sleep_time)
                #finally, go down to the ground plane and open the gripper
                if theta==0:
                    newposition[0] = newposition[0]+ 10*newposition[0]/537.0
                    newposition[1] = newposition[1]+ 10*newposition[1]/537.0
                    newposition[2] = newposition[2]-20
                newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]-30,newposition[1]+30,newposition[2],currentposition[3]]), theta)
                self.rxarm.set_positions(newwaypoint)
                rospy.sleep(sleep_time)
                self.rxarm.open_gripper()
                nstack += 1
                x = x + 0
                y = y + 0

        self.next_state = "idle"
        self.status_message = "comp4 complete, going to idle"         

    def comp5(self):
        #To the sky
        self.current_state = "comp5"
        self.status_message = "Currently Executing Competition 5"
        self.next_state = "comp5"
        self.rxarm.set_moving_time(3)
        self.rxarm.set_accel_time(.5)
        
        sleep_time = 3
        self.camera.detectBlocksInDepthImage(895,928)
        dh_params = parse_dh_param_file("./config/rx200_dh.csv")
        blocks_detections = self.camera.block_detections
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 
            size = "large" if float(block[5])>1150 else "small"
            
            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],100,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and close the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+10,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to dropoff location, 100 mm in the air
            # newposition = np.array([currentposition[0], currentposition[1], currentposition[2]+100])
            # newupwaypoint = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            # self.rxarm.set_positions(newupwaypoint)
            # rospy.sleep(sleep_time)
            #finally, go down to the ground plane and open the gripper
            # newwaypoint = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+20,currentposition[3]]))
            # self.rxarm.set_positions(newwaypoint)
            # rospy.sleep(sleep_time)
            self.rxarm.open_gripper()
            self.next_state = "idle"
            self.status_message = "pick and place complete, going to idle"

        home_pos, theta = IK_geometric(dh_params, np.array([0,88,88,0]))
        self.rxarm.set_positions(home_pos)
        rospy.sleep(sleep_time)

        self.camera.detectBlocksInDepthImage(930,960)
        blocks_detections = self.camera.block_detections
        height = 10
        for block in blocks_detections:
            block_position = np.array([float(block[1]), float(block[2]), float(block[3])])
            color, angle = block[0], float(block[4]) 

            #we haven't picked up a block yet
            self.rxarm.open_gripper()            
            self.status_message = "Moving to pick up block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            print("currentposition: ", currentposition)
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to pickup location, 100 mm in the air
            newposition = np.array([block_position[0], block_position[1], block_position[2]-10, 0])
            print("newposition:", newposition)
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and close the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+10,currentposition[3]]), theta)
            # self.rxarm.set_single_joint_position("wrist_rotate", newwaypoint[0] + np.pi/2 + -angle*180/np.pi, 1, 0.3)
            if(theta != 0):
                rotation = newwaypoint[0]*180/np.pi+angle
                rotation = int(rotation) % 90
                rotation = rotation*np.pi/180
                newwaypoint = np.append(newwaypoint, rotation)
            # print("newwaypoint:" ,newwaypoint)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.close_gripper()

            #we have picked up a block and need to place it
            self.status_message = "Moving to drop off block"
            #IK stuff to make the robot go to the location. go up, across, then down
            #get current position of end effector and go up to 100 mm
            currentposition = self.rxarm.get_ee_pose()
            upwaypoint, theta = IK_geometric(dh_params,np.array([currentposition[0],currentposition[1],currentposition[2]+300,currentposition[3]]))
            self.rxarm.set_positions(upwaypoint)
            rospy.sleep(sleep_time)
            #now go across to dropoff location, 100 mm in the air
            newposition = np.array([0, 270, height])
            newupwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+100,currentposition[3]]))
            self.rxarm.set_positions(newupwaypoint)
            rospy.sleep(sleep_time)
            #finally, go down to the ground plane and open the gripper
            newwaypoint, theta = IK_geometric(dh_params,np.array([newposition[0]+self.xoffset,newposition[1]+self.yoffset,newposition[2]+10,currentposition[3]]), theta)
            self.rxarm.set_positions(newwaypoint)
            rospy.sleep(sleep_time)
            self.rxarm.open_gripper()
            self.next_state = "idle"
            self.status_message = "pick and place complete, going to idle"
            height = height + 40
        

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)