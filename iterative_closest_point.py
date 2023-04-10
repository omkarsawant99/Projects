# -*- coding: utf-8 -*-

# Importing all dependencies

import open3d as o3d

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import copy
import plotly.graph_objects as go

"""# Functions"""

# Custom implementation of Procrustes
def procrustes(s_pts, t_pts):
  '''Performs Orthogonal Procrustes Analysis on two point sets'''

  source_centroid = np.mean(s_pts, axis=0)
  target_centroid = np.mean(t_pts, axis=0)

  Z = 0
  for i in range(len(t_pts)):
    term1 = (s_pts[i, :] - source_centroid).reshape(3,1)
    term2 = (t_pts[i, :] - target_centroid).reshape(3,1)
    Z = Z + term1 @ term2.T
  
  u, s, v_transpose = np.linalg.svd(Z)
  v = v_transpose.T
  det = np.linalg.det(v @ u.T)
  R_ts = v @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, det]]) @ u.T                  # Required rotation of source w.r.t target
  t_ts = target_centroid.reshape(3,1) - (R_ts @ source_centroid.reshape(3,1))

  R_st = R_ts.T
  t_st = -R_st @ t_ts

  H = np.hstack((R_st, t_st))
  H = np.vstack((H, [0,0,0,1]))

  return H

def findDistanceBetweenPCDs(s_pts, t_pts):
  '''Gives the distance between the centroids of two point clouds'''
  source_centroid = np.mean(s_pts, axis=0)
  target_centroid = np.mean(t_pts, axis=0)

  dist = np.linalg.norm(source_centroid - target_centroid)

  return dist

# KD Tree correspondences
def findCorrespondences(s_pts, t_pts, source_cloud):
  '''Gives approximate corresponding source points for a given set of 
  target points. For this function, the set of source points >= the set
  of target points.
  '''

  pcd_tree = o3d.geometry.KDTreeFlann(source_cloud)

  s_pts_corr = np.zeros_like(t_pts)
  for i in range(np.size(t_pts, 0)):
    [_, idx, _] = pcd_tree.search_knn_vector_3d(t_pts[i], 1)
    s_pts_corr[i, :] = s_pts[idx, :]

  return s_pts_corr

def draw_registration_result(source, target, transformation):
  """
  param: source - source point cloud
  param: target - target point cloud
  param: transformation - 4 X 4 homogeneous transformation matrix
  """
  source_temp = copy.deepcopy(source)
  target_temp = copy.deepcopy(target)
  source_temp.paint_uniform_color([1, 0.706, 0])
  target_temp.paint_uniform_color([0, 0.651, 0.929])
  source_temp.transform(transformation)
  o3d.visualization.draw_geometries([source_temp, target_temp],
                                    zoom=0.4459,
                                    front=[0.9288, -0.2951, -0.2242],
                                    lookat=[1.6784, 2.0612, 1.4451],
                                    up=[-0.3402, -0.9189, -0.1996])


"""# ICP"""

demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

s_pts = np.asarray(source.points)
t_pts = np.asarray(target.points)

# ICP
dist = 10
final_T = np.eye(4,4)
while(dist >= 0.001):
  # Finding correspondences using kd tree
  s_corr = findCorrespondences(s_pts, t_pts, source)

  # Finding distance between source and target
  dist = findDistanceBetweenPCDs(s_corr, t_pts)                       # This step is done before transforming the target cloud as correspondences change after each transformation of target cloud. 
                                                                      # Thus we would have to perform another kd tree (findCorrespondences) to find new source correspondences (s_corr) and then,
                                                                      # subtract with transformed t_pts

  # Finding transformation using procrustes
  transformation = procrustes(s_corr, t_pts)

  #Transforming target cloud
  t_pts = transformation @ np.vstack((t_pts.T, np.ones((1, t_pts.shape[0]))))
  t_pts = np.delete(t_pts, 3, 0)
  t_pts = t_pts.T

  # Final transformation
  final_T = final_T @ transformation


# Creating the new point cloud from the transformed points
end_target = o3d.geometry.PointCloud()
end_target.points = o3d.utility.Vector3dVector(t_pts)

"""# Visualizing the results"""
draw_registration_result(source, target, np.linalg.inv(final_T))
