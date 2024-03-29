# -*- coding: utf-8 -*-

import g2o
from .utils import *

class VertexEllipse(g2o.VectorXVertex):
    """A circle parameterized by position x,y with radius a,b """

    def __init__(self) -> None:
        g2o.VectorXVertex.__init__(self)
        self.set_dimension(5)
        self.set_estimate([0] * 5)

    def oplus_impl(self, update) -> None:
        self.set_estimate(self.estimate() + update)
       
class VertexTheta(g2o.VectorXVertex):
    """A angle of point on ellipse parameterized by theta """

    def __init__(self) -> None:
        g2o.VectorXVertex.__init__(self)
        self.set_dimension(1)
        self.set_estimate([0])

    def oplus_impl(self, update) -> None:
        self.set_estimate(self.estimate() + update)
    
class EdgePointOnEllipse(g2o.VariableVectorXEdge):
    def __init__(self) -> None:
        g2o.VariableVectorXEdge.__init__(self)
        self.set_dimension(1)  # dimension of the error function
        self.information()
        self.resize(2)  # number of vertices
        self.set_measurement([0, 0])  # initial measurement

    def compute_error(self):
        ellipse = self.vertex(0).estimate()
        theta = self.vertex(1).estimate()
        
        cx = ellipse[0]
        cy = ellipse[1]
        
        a = ellipse[2]
        b = ellipse[3]
        phi = ellipse[4]
        
        R = get_rot_mat_from_angle(phi)
        
        estimate = R @ np.array([a * np.cos(theta), b * np.sin(theta)]).squeeze() + np.array([cx,cy])
        
        error = np.linalg.norm(self.measurement() -  estimate)
        return [error]