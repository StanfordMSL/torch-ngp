import os
import cv2 
import json
from cv2 import transform
import numpy as np


import matplotlib.pyplot as plt 
import pprint


Rx = lambda theta: np.array([[1, 0, 0],
                             [0, np.cos(theta), -1*np.sin(theta)],
                             [0, np.sin(theta), np.cos(theta)]
                            ])
Ry = lambda phi: np.array([[np.cos(phi), 0, np.sin(phi)],
                           [0,1,0],
                           [-1*np.sin(phi), 0, np.cos(phi)]
                          ])
Rz = lambda psi: np.array([[np.cos(psi), -1*np.sin(psi),0],
                           [np.sin(psi),np.cos(psi),0],
                           [0,0,1]
                          ])


# read in robotic arm poses
path = "/home/gsznaier/torch-ngp/data/touchnerf_far/color"
with open(os.path.join(path, 'transforms_train.json'), 'r') as f:
    arm_transform = json.load(f)

arm_frames = arm_transform["frames"]
arm_w2c = []
for frame in arm_frames:
    c2w = np.linalg.inv(np.array(frame['transform_matrix'], dtype=np.float32))
    c2w[:3,:3] = Rx(np.pi)@Rz(-np.pi/2)@c2w[:3,:3]
    arm_w2c.append(np.linalg.inv(c2w))
arm_w2c = np.asarray(arm_w2c)


# read in colmap poses
path = '/home/gsznaier/Desktop/instant-ngp/data/touch_far_test'
with open(os.path.join(path, 'transforms.json'), 'r') as f:
    colmap_transform = json.load(f)

colmap_frames = colmap_transform["frames"]
colmap_w2c = []
for frame in colmap_frames:
    mat = np.array(frame['transform_matrix'], dtype=np.float32)
    print(mat)
    mat[:3,-1] = mat[:3,-1]
    
    colmap_w2c.append(mat)
colmap_w2c = np.asarray(colmap_w2c)


# set up frame axis
arm_n = arm_w2c.shape[0]
colmap_n = colmap_w2c.shape[0]

arm_origin = np.zeros((arm_n,4,1))
colmap_origin = np.zeros((colmap_n,4,1))
arm_origin[:,-1,0] = 1
colmap_origin[:,-1,0] = 1

scale = .05

e1 = np.array([[[scale],[0],[0],[1]]])
e2 = np.array([[[0],[scale],[0],[1]]])
e3 = np.array([[[0],[0],[scale],[1]]])

arm_e1 = np.repeat(e1, arm_n, axis=0)
arm_e2 = np.repeat(e2, arm_n, axis=0)
arm_e3 = np.repeat(e3, arm_n, axis=0)

colmap_e1 = np.repeat(e1, colmap_n, axis=0)
colmap_e2 = np.repeat(e2, colmap_n, axis=0)
colmap_e3 = np.repeat(e3, colmap_n, axis=0)

print(arm_w2c.shape)
print(colmap_w2c.shape)
arm_origins = (arm_w2c@arm_origin).reshape(-1,4)
arm_e1 = (arm_w2c@arm_e1).reshape(-1,4)
arm_e2 = (arm_w2c@arm_e2).reshape(-1,4)
arm_e3 = (arm_w2c@arm_e3).reshape(-1,4)

print(arm_w2c.shape)
print("colmap")
print(colmap_w2c.shape)
print("origin")

colmap_origins = (colmap_w2c@colmap_origin).reshape(-1,4)
colmap_e1 = (colmap_w2c@colmap_e1).reshape(-1,4)
colmap_e2 = (colmap_w2c@colmap_e2).reshape(-1,4)
colmap_e3 = (colmap_w2c@colmap_e3).reshape(-1,4)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(arm_origins[:,0], arm_origins[:,1], arm_origins[:,2])
ax.scatter(colmap_origin[0,0], colmap_origin[0,1], colmap_origin[0,2],'k')
for i in range(arm_n):
    ax.plot([arm_origins[i,0], arm_e1[i,0]], [arm_origins[i,1], arm_e1[i,1]], [arm_origins[i,2], arm_e1[i,2]], 'r')
    ax.plot([arm_origins[i,0], arm_e2[i,0]], [arm_origins[i,1], arm_e2[i,1]], [arm_origins[i,2], arm_e2[i,2]], 'g')
    ax.plot([arm_origins[i,0], arm_e3[i,0]], [arm_origins[i,1], arm_e3[i,1]], [arm_origins[i,2], arm_e3[i,2]], 'b')

for i in range(colmap_n):
    ax.plot([colmap_origins[i,0], colmap_e1[i,0]], [colmap_origins[i,1], colmap_e1[i,1]], [colmap_origins[i,2], colmap_e1[i,2]], 'maroon')
    ax.plot([colmap_origins[i,0], colmap_e2[i,0]], [colmap_origins[i,1], colmap_e2[i,1]], [colmap_origins[i,2], colmap_e2[i,2]], 'lime')
    ax.plot([colmap_origins[i,0], colmap_e3[i,0]], [colmap_origins[i,1], colmap_e3[i,1]], [colmap_origins[i,2], colmap_e3[i,2]], 'indigo')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
