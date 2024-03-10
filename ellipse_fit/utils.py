# -*- coding: utf-8 -*-

import numpy as np

def get_rot_mat_from_angle(phi):
    return np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])