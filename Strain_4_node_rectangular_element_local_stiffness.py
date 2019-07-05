# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 21:55:44 2019

@author: Dhayan
"""

# generation of a local stiffness matrix for 4 node 2D rectangular strain element
# the stiffness matrix generated for an isoparametric element.
 
import numpy as np

xy = np.array([[2,5,4,1],[1,2,6,4]])
xy = xy.T

E = 1.0 # Youngs modulus
nu = 0.25 #Poissons ratio

C = np.zeros([3,3])

C[0][0] = E / (1-nu**2)
C[0][1] = (E*nu) / (1-nu**2)
C[1][0] = C[0][1]
C[1][1] = C[0][0]
C[2][2] = E / (2*(1+nu))


# using two point integration formula (gauss quadrature method)
et = np.array([-0.57735, -0.57735, 0.57735, 0.57735])
ne = np.array([-0.57735, 0.57735, -0.57735, 0.57735])

# weights for Gauss quadrature points.
wi = 1
wj = 1

t = 1 #thickness of the element.



def jacobian(Mat_N, xy):
    ja = np.matmul(Mat_N, xy) 
    return ja

def rebuildB(b):
    i = np.identity(2)
    f = np.tensordot(b,i,axes=0)
    
    f0 = np.reshape(f[0],[8,2])
    f1 = np.reshape(f[1],[8,2])
    
    new_b = np.zeros([3,8])
    new_b[0,:] = f0[:,0]
    new_b[1,:] = f1[:,1]
    new_b[2,:] = f0[:,1]+f1[:,0]
    
    return new_b
    

def deformation_element_4_node_rect(xy,C,t):
    
    K = np.zeros([8,8]) #initialize a 4 by 4 matrix.
    for n in range(4):
        
        # B matrix in the eta, neta plane.
        b = np.array([[-(1-ne[n]), (1-ne[n]), (1+ne[n]), -(1+ne[n])], [-(1-et[n]), -(1+et[n]), (1+et[n]), (1-et[n])]])
        b = 0.25*b
    
        J = jacobian(b, xy)
        Jdet = np.linalg.det(J) # find determinant of jacobian
        Jinv = np.linalg.inv(J) # find inverse of jacobian
        da = Jdet*wi*wj # need to check if application of weights are accurate.
    
        B = np.matmul(Jinv, b) # transform b matrix in eta, neta to x and y plane.
        B = rebuildB(B) # convert B matrix into 3x8
        
        # local stiffness matrix.
        K1 = np.matmul(B.T, C)
        K1 = np.matmul(K1,B)
        K1 = K1*da*t
        
        K += K1
        
    return K

K = deformation_element_4_node_rect(xy,C,t)
