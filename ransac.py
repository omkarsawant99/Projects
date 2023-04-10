# -*- coding: utf-8 -*-

# Importing dependencies

import open3d as o3d

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
import copy

# Function for visualizing open3d point clouds in google colab. Couldn't install open3d in conda environment.

def draw_geometries(geometries):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()
        
        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=1, color=colors))
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)
            
            mesh_3d = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=triangles[:,0], j=triangles[:,1], k=triangles[:,2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)
        
    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.show()

o3d.visualization.draw_geometries = draw_geometries

# read demo point cloud provided by Open3D
pcd_point_cloud = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)

# function to visualize the point cloud
o3d.visualization.draw_geometries([pcd]
zoom=1,
front=[0.4257, -0.2125, -0.8795],
lookat=[2.6172, 2.0475, 1.532],
up=[-0.0694, -0.9768, 0.2024])

"""# RANSAC code"""

from google.colab import output

def runRANSACPF(pts, no_of_iterations, threshold):
  inlier_pts = []
  
  max_inlier_count = 0
  no_of_pts = pts.shape[0]
  for iter in range(no_of_iterations):
    print('Iterations:', iter)
    # Selecting 3 random points
    randpts = np.zeros((3, 3))
    for i in range(3):
      r = random.randint(0, no_of_pts - 1)
      randpts[i, :] = pts[r, :]

    p = randpts[0, :]
    q = randpts[1, :]
    r = randpts[2, :]

    # Finding equation of plane using random points
    pq = q - p
    pr = r - p

    n = np.cross(pq, pr)
    
    # Co-efficients of equation of plane : ax + by + cz = d
    #a = n[0]
    #b = n[1]
    #c = n[2]
    #d = n[0]*p[0] + n[1]*p[1] + n[2]*p[2]
    d = np.sum(n*p)

    # Running through all points to find the number of inliers (points that satisfy the equation of plane)
    inlier_count = 0
    for i in range(no_of_pts):
      d_eachpt = np.sum(n*pts[i, :])
      if abs(d_eachpt - d) < threshold:
        inlier_count = inlier_count + 1
        inlier_pts.append(pts[i, :])
    
    # Comparing the inlier count of current iteration with the previous one
    if(inlier_count > max_inlier_count):
      max_inlier_count = inlier_count
      ransac_pts = inlier_pts
      n_fitted = n
      d_fitted = d
    else:
      inlier_pts = []


  output.clear() 
  print('RANSAC finished.......')
  print('Number of inliers = ', max_inlier_count)
  #print('Iteration number = ', iter_no)
  print('Equation of fitted plane : ax + by + cz = d, where -')
  print('[a, b, c] =', n_fitted)
  print('d =', d_fitted)
  return np.array(ransac_pts) 


# Running RANSAC
points = np.asarray(pcd.points)
rnsc_pts = runRANSACPF(points, 100, 0.002)
print('Total points =', points.shape[0])

"""# Visualizing the result"""

inlier_cloud = o3d.geometry.PointCloud()
inlier_cloud.points = o3d.utility.Vector3dVector(rnsc_pts)
o3d.visualization.draw_geometries([inlier_cloud]
zoom=1,
front=[0.4257, -0.2125, -0.8795],
lookat=[2.6172, 2.0475, 1.532],
up=[-0.0694, -0.9768, 0.2024])
