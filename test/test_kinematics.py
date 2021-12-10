#!/usr/bin/python
"""!
Test kinematics

TODO: Use this file and modify as you see fit to test kinematics.py
"""
import argparse
import sys
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.realpath(script_path + '/../'))
from kinematics import *
from config_parse import *
from copy import deepcopy

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--poxconfig", required=True, help="path to pox parameters csv file")
    ap.add_argument("-p","--dhconfig", required=True, help="path to DH parameters csv file")
    args=vars(ap.parse_args())

    passed = True
    vclamp = np.vectorize(clamp)

    dh_params = parse_dh_param_file(args['dhconfig'])
    pox_params = parse_pox_param_file(args['poxconfig'])
    #*np.pi/180.0
    ### Add arm configurations to test here
    fk_angles = [29.0*np.pi/180.0, 46.0*np.pi/180.0, 47.0*np.pi/180.0, -87.0*np.pi/180.0,0.0*np.pi/180.0]
    
    print('Test FK')
    
    pose = FK_pox(fk_angles,deepcopy(pox_params)[0],deepcopy(pox_params)[1])
    print('poses: ', pose)
    fk_poses = pose
    #for joint_angles in fk_angles:
    #    print('Joint angles:', joint_angles)
    #    for i, _ in enumerate(joint_angles):
    #        pose = get_pose_from_T(FK_dh(deepcopy(dh_params), joint_angles, i))
    #        print('Link {} pose: {}'.format(i, pose))
    #        if i == len(joint_angles) - 1:
    #            fk_poses.append(pose)
    #    print()

    print('Test IK')
    options = IK_geometric(deepcopy(dh_params), fk_poses)
    print('IK: ', options)
    
    
    #for pose, angles in zip(fk_poses, fk_angles):
    #    matching_angles = False
    #
#        print('Pose: {}'.format(pose))
#        options = IK_geometric(deepcopy(dh_params), fk_poses)
#        for i, joint_angles in enumerate(options):
#            print('Option {}: {}'.format(i, joint_angles))
#            compare = vclamp(joint_angles - angles)
#            if np.allclose(compare, np.zeros_like(compare), rtol=1e-3, atol=1e-4):
#                print('Option {} matches angles used in FK'.format(i))
#                matching_angles = True
#        if not matching_angles:
#            print('No match to the FK angles found!')
#            passed = False
#        print()
