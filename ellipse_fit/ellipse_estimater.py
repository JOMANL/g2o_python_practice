# -*- coding: utf-8 -*-

import numpy as np
from .utils import *

def est_ellipse_from_points(points):
    """ estimate shape parameter and angle of each sample by SVD

    Args:
        points (ndarray): N x 2 matrix for point coordinate

    Returns:
        est_angles : estimated sample's angle from long axis
        est_cx : estimated x coordinate of center
        est_cy : estimated y coordinate of center
        est_A : estimated long axis length
        est_B : estimated short axis length
        est_rotated_angle : estimated rotated angle of long axis from x axis
    """

    if not isinstance(points, np.ndarray):
        raise TypeError("points argument must be a NumPy ndarray")

    if points.shape[1] != 2:
        raise ValueError("points argument must have a shape of (N, 2)")
    
    est_cx = points[:,0].mean()
    est_cy = points[:,1].mean()
    
    points_sifted= (np.array(points) - np.array([est_cx,est_cy])).T
    U,S,V = np.linalg.svd(points_sifted)
    est_rotated_angle = np.arctan2(U[1][0], U[0][0])
    
    est_rot = get_rot_mat_from_angle(est_rotated_angle)
    
    axis_ratio = S[0] / S[1]

    points_rotated = (est_rot.T @ points_sifted).T

    est_angles = [np.arctan2(axis_ratio * y,x) for x,y in zip(points_rotated[:,0],points_rotated[:,1])]
    
    long_axs = []
    short_axs = []
    
    for angle,point in zip(est_angles,points_rotated):
        long_axs.append(point[0] / np.cos(angle))
        short_axs.append(point[1] / np.sin(angle))
        
    est_A = np.array(long_axs).mean()
    est_B = np.array(short_axs).mean()
       
    return est_angles,est_cx,est_cy,est_A,est_B,est_rotated_angle