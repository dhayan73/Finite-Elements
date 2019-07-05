# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:07:01 2019

@author: Dhayan
"""

import numpy as np

xy = np.array([[2,5,4,1],[1,2,6,4]])
xy = xy.T

k = np.array([[1,0],[0,1]]) # thermal coefficients.

# using two point integration formula (Gauss quadrature method)
et = np.array([-0.57735, -0.57735, 0.57735, 0.57735])
ne = np.array([-0.57735, 0.57735, -0.57735, 0.57735])

# weights for Gauss quadrature points.
wi = 1
wj = 1

t = 1 # element thickness.

def jacobian(Mat_N, xy):
    ja = np.matmul(Mat_N, xy) 
    return ja

def thermal_element_4_node_rect(xy,k,t):
    
    K = np.zeros([4,4]) #initialize a 4 by 4 matrix.
    for n in range(4):
        
        # B matrix in the eta, neta plane.
        b = np.array([[-(1-ne[n]), (1-ne[n]), (1+ne[n]), -(1+ne[n])], [-(1-et[n]), -(1+et[n]), (1+et[n]), (1-et[n])]])
        b = 0.25*b
    
        J = jacobian(b, xy)
        Jdet = np.linalg.det(J) # find determinant of jacobian
        Jinv = np.linalg.inv(J) # find inverse of jacobian
        da = Jdet*wi*wj # need to check if application of weights are accurate.
    
        B = np.matmul(Jinv, b) # transform b matrix in eta, neta to x and y plane.
        
        # local stiffness matrix.
        K1 = np.matmul(B.T, k)
        K1 = np.matmul(K1,B)
        K1 = K1*da*t
        
        K += K1
        
    return K

K = thermal_element_4_node_rect(xy,k,t)
