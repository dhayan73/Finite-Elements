# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:27:50 2019

@author: Dhayan
"""
# for axi-symmetric element.
import numpy as np

class Element:
    
    #def __init__(self, id_num, node1, node2, node3, node4, xy):
    def __init__(self, id_num, nodeList, xy):
        self.id_num = id_num
        self.node_id = np.zeros([4,1])
        self.node_id = np.ravel(self.node_id)
        self.node_id[0] = nodeList[0]
        self.node_id[1] = nodeList[1]
        self.node_id[2] = nodeList[2]
        self.node_id[3] = nodeList[3]
        self.kblocks = {}
        
        # Guass points.
        self.eta = np.array([-0.57735, -0.57735, 0.57735, 0.57735])
        self.neta = np.array([-0.57735, 0.57735, -0.57735, 0.57735])
                        
        # Geometry and material properties.
        self.xy = xy #x,y co-ordinates of the nodes.
        self.t = 1  #thickness of the element. Ususally set to 1.
        self.E = 1 #youngs modulus of the element.
        self.nu = 0.3 # Poisson ratio of the element.
        self.C = self.buildC(self.E, self.nu)
        self.k = self.calculate_stiffness(self.xy, self.C, self.t)
        
        # section local k to blocks to map and help build Global matrix.
        for r in range(4):
            for c in range(4):
                s = str(int(self.node_id[r])) + ',' + str(int(self.node_id[c]))
                self.kblocks[s] = self.k[0+2*r:2+2*r, 0+2*c:2+2*c]
                
        
        
        
    def buildC(self,E,nu):
        C = np.zeros([4,4])
        coeff = E/((1+nu)*(1-2*nu))

        C[0][0] = (1-nu)
        C[0][1] = nu
        C[0][2] = nu
        C[1][0] = nu
        C[1][1] = C[0][0]
        C[1][2] = nu
        C[2][0] = nu
        C[2][1] = nu
        C[2][2] = C[0][0]
        C[3][3] = (1-2*nu)/2
        C = coeff*C
        return C
    
    def fourNlinearShapeFunc(self,et,ne):
        # shape/ interpolation function for 4 node element
        N = np.array([[(1-et)*(1-ne), (1+et)*(1-ne), (1+et)*(1+ne), (1-et)*(1+ne)],
                       [(1-et)*(1-ne), (1+et)*(1-ne), (1+et)*(1+ne), (1-et)*(1+ne)]])
        N = 0.25*N
        return N
    
    def getGausspoints(self,et,ne,xy):
        r = 0.25*((1-et)*(1-ne)*xy[0,0] + (1+et)*(1-ne)*xy[1,0] + (1+et)*(1+ne)*xy[2,0] + (1-et)*(1+ne)*xy[3,0])
        z = 0.25*((1-et)*(1-ne)*xy[0,1] + (1+et)*(1-ne)*xy[1,1] + (1+et)*(1+ne)*xy[2,1] + (1-et)*(1+ne)*xy[3,1])
        return r,z
    
    def generateBMat(self,et,ne):
        b = np.array([[-(1-ne), (1-ne), (1+ne), -(1+ne)], [-(1-et), -(1+et), (1+et), (1-et)]])
        b = 0.25*b
        return b
        
    def jacobian(self,Mat_N, xy):
        ja = np.matmul(Mat_N, xy) 
        return ja

    def rebuildB(self,b, N, r):
        i = np.identity(2)
        f = np.tensordot(b,i,axes=0)
        
        f0 = np.reshape(f[0],[8,2])
        f2 = np.reshape(f[1],[8,2])
        
        f1 = N/r
        
        new_b = np.zeros([4,8])
        new_b[0,:] = f0[:,0]
        new_b[2,:] = f2[:,1]
        new_b[3,:] = f0[:,1]+f2[:,0]
        new_b[1,0] = f1[0,0]
        new_b[1,2] = f1[0,1]
        new_b[1,4] = f1[0,2]
        new_b[1,6] = f1[0,3]
        
        return new_b
        
    def calculate_stiffness(self,xy,C,t):
        
        # using two point integration formula (gauss quadrature method)
        #et = np.array([-0.57735, -0.57735, 0.57735, 0.57735])
        #ne = np.array([-0.57735, 0.57735, -0.57735, 0.57735])
        
        et = self.eta
        ne = self.neta

        wi = 1
        wj = 1

        #t = 1 #thickness of the element.
        
        K = np.zeros([8,8]) #initialize a 4 by 4 matrix.
        for n in range(4):
            
            # shape/ interpolation function for 4 node element
            #N = np.array([[(1-et[n])*(1-ne[n]), (1+et[n])*(1-ne[n]), (1+et[n])*(1+ne[n]), (1-et[n])*(1+ne[n])],
            #           [(1-et[n])*(1-ne[n]), (1+et[n])*(1-ne[n]), (1+et[n])*(1+ne[n]), (1-et[n])*(1+ne[n])]])
            #N = 0.25*N
            N = self.fourNlinearShapeFunc(et[n],ne[n])
            
            # calculate r
            #r = 0.25*((1-et[n])*(1-ne[n])*xy[0,0] + (1+et[n])*(1-ne[n])*xy[1,0] + (1+et[n])*(1+ne[n])*xy[2,0] + (1-et[n])*(1+ne[n])*xy[3,0])
            
            # calculate z
            #z = 0.25*((1-et[n])*(1-ne[n])*xy[0,1] + (1+et[n])*(1-ne[n])*xy[1,1] + (1+et[n])*(1+ne[n])*xy[2,1] + (1-et[n])*(1+ne[n])*xy[3,1])
            r,z = self.getGausspoints(et[n],ne[n],xy)
            
            # B matrix in the eta, neta plane.
            #b = np.array([[-(1-ne[n]), (1-ne[n]), (1+ne[n]), -(1+ne[n])], [-(1-et[n]), -(1+et[n]), (1+et[n]), (1-et[n])]])
            #b = 0.25*b
            b = self.generateBMat(et[n],ne[n])
        
            J = self.jacobian(b, xy)
            Jdet = np.linalg.det(J) # find determinant of jacobian
            Jinv = np.linalg.inv(J) # find inverse of jacobian
            da = 2*np.pi*r*Jdet*wi*wj # need to check if application of weights are accurate.
        
            B = np.matmul(Jinv, b) # transform b matrix in eta, neta to r and z plane.
            B = self.rebuildB(B, N, r) # convert B matrix into 3x8
            
            # local stiffness matrix.
            K1 = np.matmul(B.T, C)
            K1 = np.matmul(K1,B)
            K1 = K1*da*t
            
            K += K1
            
        return K
        
# number of elements
num_elem = 5

# connectivity matrix which in this case can be a dictionary.
connect_mat = {}
for n in range(num_elem):
    connect_mat[n] = [1,2,3,4]

connect_mat[1] = [2,5,6,3]
connect_mat[2] = [5,7,8,6]
connect_mat[3] = [7,9,10,8]
connect_mat[4] = [9,11,12,10]

#connect_mat[5] = [11,13,14,12]
#connect_mat[6] = [13,15,16,14]
#connect_mat[7] = [15,17,18,16]
#connect_mat[8] = [17,19,20,18]
#connect_mat[9] = [19,21,22,20]


# nodal point positions in xy matrix
rz = {}
pts1 = np.array([[1.0,1.2,1.2,1.0],[0.0,0.0,0.2,0.2]])
pts2 = np.array([[1.2,1.4,1.4,1.2],[0.0,0.0,0.2,0.2]])
pts3 = np.array([[1.4,1.6,1.6,1.4],[0.0,0.0,0.2,0.2]])
pts4 = np.array([[1.6,1.8,1.8,1.6],[0.0,0.0,0.2,0.2]])
pts5 = np.array([[1.8,2.0,2.0,1.8],[0.0,0.0,0.2,0.2]])

#pts1 = np.array([[1.0, 1.1, 1.1, 1.0], [0.0,0.0,0.1,0.1]])
#pts2 = np.array([[1.1, 1.2, 1.2, 1.1], [0.0,0.0,0.1,0.1]])
#pts3 = np.array([[1.2, 1.3, 1.3, 1.2], [0.0,0.0,0.1,0.1]])
#pts4 = np.array([[1.3, 1.4, 1.4, 1.3], [0.0,0.0,0.1,0.1]])
#pts5 = np.array([[1.4, 1.5, 1.5, 1.4], [0.0,0.0,0.1,0.1]])
#pts6 = np.array([[1.5, 1.6, 1.6, 1.5], [0.0,0.0,0.1,0.1]])
#pts7 = np.array([[1.6, 1.7, 1.7, 1.6], [0.0,0.0,0.1,0.1]])
#pts8 = np.array([[1.7, 1.8, 1.8, 1.7], [0.0,0.0,0.1,0.1]])
#pts9 = np.array([[1.8, 1.9, 1.9, 1.8], [0.0,0.0,0.1,0.1]])
#pts10 = np.array([[1.9, 2.0, 2.0, 1.9], [0.0,0.0,0.1,0.1]])

rz[0] = pts1.T
rz[1] = pts2.T
rz[2] = pts3.T
rz[3] = pts4.T
rz[4] = pts5.T

#rz[5] = pts6.T
#rz[6] = pts7.T
#rz[7] = pts8.T
#rz[8] = pts9.T
#rz[9] = pts10.T

# initialise the elements
e = {}
for el in range(num_elem):
    e[el] = Element(el+1, connect_mat[el], rz[el])


# build global stiffness matrix
K = np.zeros([24,24])

#K = np.zeros([44,44])

for nel in range(num_elem):
    kb = e[nel].kblocks
    for n in kb.keys():
        
        idx = n.split(',')
        r = int(idx[0])-1
        c = int(idx[1])-1
        
        K[0+r*2: 2+r*2, 0+c*2: 2+c*2] += kb[n]     
        
L = K.copy()
L = np.delete(L, [1,3,5,7,9,11,13,15,17,19,21,23], axis = 0)
L = np.delete(L, [1,3,5,7,9,11,13,15,17,19,21,23], axis = 1)

#L = np.delete(L, [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43], axis = 0)
#L = np.delete(L, [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43], axis = 1)

#f = np.zeros([24,1])
#f[0] = 628.32
#f[6] = 628.32

#f = np.zeros([44,1])
#f[0] = 314.16
#f[6] = 314.16


#fL = np.delete(f, [1,3,5,7,9,11,13,15,17,19,21,23], axis = 0)
#fL = np.delete(f, [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43], axis = 0)

# pressure defined in terms of pounds per square inch.
px = 1000 #(pounds per square inch)

# need to find dx and dy. # if load is applied to nodes 1 and 4 on element 1, then eta = -1,
# so substitute for neta the two Gauss points.

eta = -1 
neta = -0.57735
b1 = e[0].generateBMat(eta,neta)

# check the calculations and remember that d/deta = 0, d/dneta corresponds to 2nd row of the jacobian.
# to find the dx and dy we matmul with the x and y co-ordinates of this element.
x = e[0].xy[:,0]
x = np.reshape(x,[4,1])
y = e[0].xy[:,1]
y = np.reshape(y,[4,1])
dx = np.matmul(b1[1,:], x)
dy = np.matmul(b1[1,:], y)

# get dL
dL = (dx**2 + dy**2)**0.5

# get the shape function when eta = -1 and neta = -0.57735
N = e[0].fourNlinearShapeFunc(eta,neta)

# load distribution on node 1.
l1 = N[0,0]*px*dL*2*np.pi
# load contribution on node 4.
l4 = N[0,3]*px*dL*2*np.pi

# get shape function when eta=-1 and neta = 0.57735
neta = 0.57735
N = e[0].fourNlinearShapeFunc(eta,neta)

# load distribution for this Gauss point
l11 = N[0,0]*px*dL*2*np.pi
l44 = N[0,3]*px*dL*2*np.pi

# total load acting on each point.

load_1 = l1+l11
load_4 = l4+l44
 

#L_inv = np.linalg.inv(L)

# calculate the deflections.
#u = np.matmul(L_inv, fL)



'''
# example to calculate stress for element 1 for all four Guass points.
et = e[0].eta
ne = e[0].neta

b1 = e[0].generateBMat(et[0],ne[0])
b2 = e[0].generateBMat(et[1],ne[1])
b3 = e[0].generateBMat(et[2],ne[2])
b4 = e[0].generateBMat(et[3],ne[3])



xy1 = e[0].xy
J1 = e[0].jacobian(b1, xy1)
J1Inv = np.linalg.inv(J1)
pB1 = np.matmul(J1Inv, b1)


N1 = e[0].fourNlinearShapeFunc(et[0],ne[0])
r1,z1 = e[0].getGausspoints(et[0],ne[0],xy1)
B1 = e[0].rebuildB(pB1, N1, r1)
C = e[0].C

u1 = np.array([[1894.35, 0, 1642.63, 0, 1642.63, 0, 1894.35, 0]])
# calculate the strain
ep1 = np.matmul(B1,u1.T) 

s1 = np.matmul(C, ep1)

J3 = e[0].jacobian(b3, xy1)
J3Inv = np.linalg.inv(J3)
pB3 = np.matmul(J3Inv, b3)


N3 = e[0].fourNlinearShapeFunc(et[2],ne[2])
r3,z3 = e[0].getGausspoints(et[2],ne[2],xy1)
B3 = e[0].rebuildB(pB3, N3, r3)

ep3 = np.matmul(B3,u1.T) 

s3 = np.matmul(C, ep3)
'''
