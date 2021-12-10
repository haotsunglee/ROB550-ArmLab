"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
import math
from scipy.spatial.transform import Rotation as ST


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass

def get_transform_from_DH(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    pass


def get_euler_angles_from_T(T,form):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    T = np.array(T)
    R = T[0:3, 0:3]
    dcm = ST.from_dcm(R)
    euler_angles = dcm.as_euler(form)
    return np.array(euler_angles)

    


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    T = np.array(T)
    #print('elr: ',get_euler_angles_from_T(T,'ZYX'))
    phi = get_euler_angles_from_T(T,'ZYX')[1]
    #print('phi: ',phi)
    #pose = np.vstack((T[0:-1,-1],phi))
    pose = T[:,-1]
    pose[-1] = phi
    #print('p: ',pose)
    return pose


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    #print('m: ',m_mat)
    #print('s: ', s_lst)
    T_tot = np.eye(4)
    num_joint = len(s_lst)
    #print('num_j',num_joint)
    joint_angles = np.array(joint_angles)
    for i in range(num_joint):
        s_vector = np.array(np.array(s_lst)[i,:])
        w = s_vector[0:3]
        v = s_vector[3:]
        #print('w: ',w)
        #print('v: ',v)
        S_mat = to_s_matrix(w,v)
        T_mat = to_T_matrix(S_mat,joint_angles[i])
        #print('theta: ',joint_angles[i],' ',i,' T: ',T_mat)
        T_tot = np.dot(T_tot,T_mat)
    T_tot = np.dot(T_tot,m_mat)
    xyzphi = get_pose_from_T(T_tot)

    return xyzphi.reshape(1, -1)[0].tolist()



def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { [w1,w2,w3] rotation degree of freedom at joint }
    @param      v     { [v1,v2,v3] corresponding linear speed at the origin}

    @return     { S matrix }
    """
    w_matrix = np.array([[0, -w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    v_np = np.array(v).reshape(-1,1)
    #print('w_matrix',w_matrix)
    #print('vnp: ',v_np)
    S_mat = np.hstack((w_matrix,v_np))
    S_mat = np.vstack((S_mat,np.array([0,0,0,0])))
    #print('S_mat',S_mat)
    return S_mat

def to_T_matrix(S_mat,joint_angle):
    #print('S_mat',S_mat)
    joint_angle = clamp(joint_angle)
    w_mat = S_mat[0:3,0:3]
    v_vec = S_mat[0:3,-1].reshape(-1,1)
    #print('v: ',v_vec)
    R = expm(joint_angle*w_mat)
    #R_r = (np.eye(3)+(math.sin(joint_angle))*w_mat+(1-math.cos(joint_angle))*(np.dot(w_mat,w_mat)))
    #print('R: ',R)
    #print('R_r',R_r)
    p =np.matmul((np.eye(3)*joint_angle+(1-math.cos(joint_angle))*w_mat+(joint_angle-math.sin(joint_angle))*(np.matmul(w_mat,w_mat))),v_vec)
    #print('p: ',p)
    T = np.hstack((R,p))
    T = np.vstack((T,np.array([0,0,0,1])))
    #print('T_mat',T)
    return T



def IK_geometric(dh_params,pose,phii = 90):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    @return     elbow up and elbow down config
    @return     None if not possible
    """
    dh_params = np.array(dh_params)
    dh_d = dh_params[:,2] 
    dh_a = dh_params[:,0]
    x,y,z,phi = pose
    all_config = np.zeros([4,2])
    theta_1 = 0 if abs(x) <= 1e-2 and abs(y) <= 1e-2 else math.atan2(x,y)
    all_config[0] = [-theta_1,-theta_1]
    R_xy = math.sqrt(x*x+y*y)
    x_c,y_c,z_c = x,y,z+dh_d[-1]            
    #if efi_ori change in the future, change this,z + sin, xcphictheta1,ycphisintheta1
    phi = phii*np.pi/180
    l_4 = dh_d[-1]
    z_c = z + math.sin(phi)*l_4
    x_c = x - math.cos(phi)*math.cos(np.pi/2-theta_1)*l_4
    y_c = y - math.cos(phi)*math.sin(np.pi/2-theta_1)*l_4
    off_set = math.atan2(dh_a[2],dh_a[3])
    #print(x,y,z,x_c,y_c,z_c)
    l_2 = dh_a[3]
    d = dh_a[2]
    l_3 = dh_a[-1]
    l_2_p = 205.73
    v2h_offset = 40.0
    #if phii == 0:
    #    x,y,z = x+v2h_offset/math.sqrt(x*x+y*y)*x,y+v2h_offset/math.sqrt(x*x+y*y)*y,z-40.0
    # print('cos: ',(x_c*x_c+y_c*y_c+(z_c-dh_d[1])*(z_c-dh_d[1])-l_2_p*l_2_p-l_3*l_3)/(2*l_2_p*l_3))
    try:
        theta_3 = math.acos((x_c*x_c+y_c*y_c+(z_c-dh_d[1])*(z_c-dh_d[1])-l_2_p*l_2_p-l_3*l_3)/(2*l_2_p*l_3))
    except ValueError:
        print("IK pose",phii, " degree invalid!")
        if phii == 0:
            return IK_geometric(dh_params, np.array([0,88,88,0]))
        #print("IK pose: ",pose)
        #print('cos: ',(x_c*x_c+y_c*y_c+(z_c-dh_d[1])*(z_c-dh_d[1])-l_2_p*l_2_p-l_3*l_3)/(2*l_2_p*l_3))
        
        return IK_geometric(dh_params, pose, 0)
    
    
    beta = math.atan2(z_c-dh_d[1],R_xy)
    alpha = math.atan2(l_3*math.sin(theta_3),l_2_p+l_3*math.cos(theta_3))
    #print('alpha: ',alpha*180/np.pi)
    #print('beta: ',beta*180/np.pi)
    #elbow down
    theta_2_d = beta+alpha
    #theta_2_p = math.acos((-x_c*x_c-y_c*y_c-(z_c-dh_d[1])*(z_c-dh_d[1])-l_2_p*l_2_p+l_3*l_3)/(2*l_2_p*math.sqrt(x_c*x_c+y_c*y_c+(z_c-dh_d[1])*(z_c-dh_d[1]))))
    #theta_2_p = np.pi-theta_2_p if theta_2_p > np.pi/2 else theta_2_p
    #theta_2 = np.pi/2-(theta_2_p+off_set+alpha)
    if phii == 90:
        theta_4_offset = 5.0*np.pi/180.0
    elif phii == 0:
        theta_4_offset = 5*np.pi/180
    else:
        theta_4_offset = 0
    #phi = np.pi/4
    all_config[1,0] = np.pi/2-theta_2_d-off_set
    all_config[2,0] = np.pi/2-off_set-theta_3
    theta_4 = -phi - all_config[2,0] + all_config[1,0] + theta_4_offset
    all_config[3,0] = theta_4
    #elbow up
    theta_2_u = beta-alpha
    all_config[1,1] = np.pi/2 - theta_2_u - off_set
    all_config[2,1] = np.pi/2 - off_set + theta_3
    theta_4_u = -phi - all_config[2,1] + all_config[1,1]
    all_config[3,1] = theta_4_u
    #print('a: ',all_config*180/np.pi)

    return all_config[:,0],phii
