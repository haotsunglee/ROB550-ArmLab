import cv2
import numpy as np

images_points = np.array([(462,299),(933,294),(938,579),(464,582)], dtype="double")
# world_points = np.array([(-0.201978542081, 0.239866636934, 0.944745368953),(0.294953294182, 0.236562270145,0.95045311034),
# (0.28763453249, -0.0621969258088,0.944676932073),(-0.202058214896,-0.0565895974236,0.940959497487)])
world_points = np.array([(-5,5.5,0),(5,5.5,0),(5,-0.5,0),(-5,-0.5,0)])
world_points = world_points * 50

intrinsic = np.array([[930.968269,0,641.5526133],[0,927.8618057,358.2265057],[0,0,1]])
dist_coeffs = np.array([0.124025,-0.183703,-0.0001553333333,-0.003647,0])
success, rotation_vector, translation_vector = cv2.solvePnP(world_points, images_points, intrinsic, dist_coeffs, flags=0)
print("rotation : ")
# print(rotation_vector)
rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
print(rotation_matrix)
print("translation : ")
print(translation_vector)
