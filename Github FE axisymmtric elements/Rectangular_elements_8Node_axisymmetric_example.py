# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:07:36 2019

@author: Dhayan
"""
# for axi-symmetric element. 8- Node rectangular.
import numpy as np

class Element:
    
    def __init__(self, id_num, nodeList, xy):
        self.id_num = id_num
        
        # **Needs to be generalised for any number of nodes. 
        self.node_id = np.zeros([8,1])
        self.node_id = np.ravel(self.node_id)
        self.node_id[0] = nodeList[0]
        self.node_id[1] = nodeList[1]
        self.node_id[2] = nodeList[2]
        self.node_id[3] = nodeList[3]
        self.node_id[4] = nodeList[4]
        self.node_id[5] = nodeList[5]
        self.node_id[6] = nodeList[6]
        self.node_id[7] = nodeList[7]
        self.dof = 2
        self.kblocks = {}
        
        # Guass points. (2x2 Gauss quadrature points)
        #self.eta = np.array([-0.57735, -0.57735, 0.57735, 0.57735])
        #self.neta = np.array([-0.57735, 0.57735, -0.57735, 0.57735])
        #self.weights = np.array([[1,1,1,1],[1,1,1,1]])
        
        # Gauss points. (3x3 Gauss quadrature points)
        self.eta = np.array([-(0.6)**0.5, -(0.6)**0.5, -(0.6)**0.5, 0, 0, 0, (0.6)**0.5, (0.6)**0.5, (0.6)**0.5])
        self.neta = np.array([-(0.6)**0.5, 0, (0.6)**0.5, -(0.6)**0.5, 0, (0.6)**0.5, -(0.6)**0.5, 0, (0.6)**0.5])
        self.weights = np.array([[5/9, 5/9, 5/9, 8/9, 8/9, 8/9, 5/9, 5/9, 5/9],[5/9, 8/9, 5/9, 5/9, 8/9, 5/9, 5/9, 8/9, 5/9]])
                        
        # Geometry and material properties.
        self.xy = xy #x,y co-ordinates of the nodes.
        self.t = 1  #thickness of the element. Ususally set to 1.
        self.E = 1 #youngs modulus of the element.
        self.nu = 0.3 # Poisson ratio of the element.
        self.C = self.buildC(self.E, self.nu)
        self.k = self.calculate_stiffness(self.xy, self.C, self.t)
        
        # section local k to blocks to map and help build Global matrix.
        for r in range(8):
            for c in range(8):
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
        N = np.array([[(1-et)*(1-ne)*(-et-ne-1), (1+et)*(1-ne)*(et-ne-1), (1+et)*(1+ne)*(et+ne-1), (1-et)*(1+ne)*(-et+ne-1), 2*(1-et**2)*(1-ne), 2*(1+et)*(1-ne**2), 2*(1-et**2)*(1+ne), 2*(1-et)*(1-ne**2)],
                      [(1-et)*(1-ne)*(-et-ne-1), (1+et)*(1-ne)*(et-ne-1), (1+et)*(1+ne)*(et+ne-1), (1-et)*(1+ne)*(-et+ne-1), 2*(1-et**2)*(1-ne), 2*(1+et)*(1-ne**2), 2*(1-et**2)*(1+ne), 2*(1-et)*(1-ne**2)]])
        N = 0.25*N
        return N
    
    def getGausspoints(self,et,ne,xy):
        r = 0.25*((1-et)*(1-ne)*(-et-ne-1)*xy[0,0] + (1+et)*(1-ne)*(et-ne-1)*xy[1,0] + (1+et)*(1+ne)*(et+ne-1)*xy[2,0] + (1-et)*(1+ne)*(-et+ne-1)*xy[3,0] + 2*(1-et**2)*(1-ne)*xy[4,0] + 2*(1+et)*(1-ne**2)*xy[5,0] + 2*(1-et**2)*(1+ne)*xy[6,0] + 2*(1-et)*(1-ne**2)*xy[7,0])
        z = 0.25*((1-et)*(1-ne)*(-et-ne-1)*xy[0,1] + (1+et)*(1-ne)*(et-ne-1)*xy[1,1] + (1+et)*(1+ne)*(et+ne-1)*xy[2,1] + (1-et)*(1+ne)*(-et+ne-1)*xy[3,1] + 2*(1-et**2)*(1-ne)*xy[4,1] + 2*(1+et)*(1-ne**2)*xy[5,1] + 2*(1-et**2)*(1+ne)*xy[6,1] + 2*(1-et)*(1-ne**2)*xy[7,1])
        return r,z
    
    def generateBMat(self,et,ne):
        b = np.array([[(2*et+ne)*(1-ne), (2*et-ne)*(1-ne), (2*et+ne)*(1+ne), (2*et-ne)*(1+ne), -4*et*(1-ne), 2*(1-ne**2), -4*et*(1+ne), -2*(1-ne**2)], 
                      [(2*ne+et)*(1-et), (2*ne-et)*(1+et), (2*ne+et)*(1+et), (2*ne-et)*(1-et), -2*(1-et**2), -4*ne*(1+et), 2*(1-et**2), -4*ne*(1-et)]])
        b = 0.25*b
        return b
        
    def jacobian(self,Mat_N, xy):
        ja = np.matmul(Mat_N, xy) 
        return ja

    def rebuildB(self,b, N, r):
        i = np.identity(2)
        f = np.tensordot(b,i,axes=0)
        
        f0 = np.reshape(f[0],[16,2])
        f2 = np.reshape(f[1],[16,2])
        
        f1 = N/r
        
        new_b = np.zeros([4,16])
        new_b[0,:] = f0[:,0]
        new_b[2,:] = f2[:,1]
        new_b[3,:] = f0[:,1]+f2[:,0]
        new_b[1,0] = f1[0,0]
        new_b[1,2] = f1[0,1]
        new_b[1,4] = f1[0,2]
        new_b[1,6] = f1[0,3]
        new_b[1,8] = f1[0,4]
        new_b[1,10] = f1[0,5]
        new_b[1,12] = f1[0,6]
        new_b[1,14] = f1[0,7]
        
        return new_b
        
    def calculate_stiffness(self,xy,C,t):
        
        # using two point integration formula (gauss quadrature method)        
        et = self.eta
        ne = self.neta

        #initialize a 4 by 4 matrix.
        K = np.zeros([16,16]) 
        
        for n in range(np.size(et)):
            
            wi = self.weights[0,n]
            wj = self.weights[1,n]
            
            # shape/ interpolation function for 4 node element
            N = self.fourNlinearShapeFunc(et[n],ne[n])
            
            # calculate r and z
            r,z = self.getGausspoints(et[n],ne[n],xy)
            
            # B matrix in the eta, neta plane.
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
    
    
    def calculateStress(self, u):
        
        et = self.eta
        ne = self.neta
        
        xy = self.xy
        stress = np.zeros([4,np.size(et)])
        radius = np.zeros([1,np.size(et)])
        
        for n in range(np.size(et)):
            
            b = self.generateBMat(et[n],ne[n])
            J = self.jacobian(b, xy)
            JInv = np.linalg.inv(J)
            pB = np.matmul(JInv, b)
            
            N = self.fourNlinearShapeFunc(et[n],ne[n])
            r,z = self.getGausspoints(et[n],ne[n],xy)
            B = self.rebuildB(pB, N, r)
            C = self.C
            
            # calculate the strain
            epsil = np.matmul(B,u)
            # calculate stress.
            stress[:,n] = np.matmul(C, epsil)
            # record r for Gauss point.
            radius[0,n] = r
            
        return stress, radius
            
            


## ___________________________________SET UP___________________________________ 

'''

261.8 -->  4---7---3---13---10
           |       |        |
1047.2 --> 8   I   6   II   12
           |       |        |
261.8 -->  1---5---2---11---9

 
 v/y/neta
 ^
 |
 .--> u/x/eta


 As seen above for a two element 8 node rectangular strip set up above.
 Nodes 1,4 and 8 are loaded by an external load that corresponds to 1000psi.
 (each loading value is in lb)
 
 The elements are 0.25 (z) x 0.5 (r) (in2) elements.
  
 The inner radius is 1 inches 
 The outer radius is 2 inches
'''

        
# number of elements
num_elem = 2

# connectivity matrix which in this case can be a dictionary.
connect_mat = {}
for n in range(num_elem):
    connect_mat[n] = [1,2,3,4,5,6,7,8]

connect_mat[1] = [2,9,10,3,11,12,13,6]


# nodal point positions in xy matrix. (Data arranged according to ordering of Nodes)
rz = {}
pts1 = np.array([[1.00, 1.50, 1.50, 1.00, 1.25, 1.50, 1.25, 1.00],[0.00, 0.00, 0.25, 0.25, 0.00, 0.125, 0.25, 0.125]])
pts2 = np.array([[1.50, 2.00, 2.00, 1.50, 1.75, 2.00, 1.75, 1.50],[0.00, 0.00, 0.25, 0.25, 0.00, 0.125, 0.25, 0.125]])

rz[0] = pts1.T
rz[1] = pts2.T

# initialise the elements
e = {}
for el in range(num_elem):
    e[el] = Element(el+1, connect_mat[el], rz[el])

# find maximum node number. (can use for defining Global stiffness matrix size, and d.o.f)
for nn in range(len(connect_mat)):
    max_nn = 0
    mn = max(connect_mat[nn])
    if mn > max_nn:
        max_nn = mn


# build global stiffness matrix (max_nn * d.o.f)
K = np.zeros([max_nn*2, max_nn*2])

for nel in range(num_elem):
    kb = e[nel].kblocks
    for n in kb.keys():
        
        idx = n.split(',')
        r = int(idx[0])-1
        c = int(idx[1])-1
        
        K[0+r*2: 2+r*2, 0+c*2: 2+c*2] += kb[n]     
        

# define load matrix as a set of zeros.
F = np.zeros([max_nn*2,1])

# start with a displacement map {NODE: [index]}
disp_map = {1:[0,1],
            2:[2,3],
            3:[4,5],
            4:[6,7],
            5:[8,9],
            6:[10,11],
            7:[12,13],
            8:[14,15],
            9:[16,17],
            10:[18,19],
            11:[20,21],
            12:[22,23],
            13:[24,25]}



# list of boundary conditions.

# all the y (v) directional displacements are zero.
BCS = {1:0, 3:0, 5:0, 7:0, 9:0, 11:0, 13:0, 15:0, 17:0, 19:0, 21:0, 23:0, 25:0}

# Apply boundary conditions to the stiffness and external load matrices.
for bc in BCS.keys():
    offset_fmat = K[:,bc]*BCS[bc] # calculate amount to adjust the external load matrix when BC applied to L
    offset_fmat = np.reshape(offset_fmat,[max_nn*2,1])
    F = F - offset_fmat # re-set the load matrix.
    F[bc] = BCS[bc] # set the BC value to the relevant load parameter.
    
    # Apply BCs to the stiffness matrix.
    K[:,bc] = 0
    K[bc,:] = 0
    K[bc,bc]= 1

# nodes 1, 4 and 8 (correspond to indexes 0, 6 and 14 of the F and K matrices) have 
# external loads applied to them. 
F[0] = 261.8
F[6] = 261.8
F[14] = 1047.20


# Calculate the deformations.
Kinv = np.linalg.inv(K)

# Total nodal displacement for entire mesh.
u = np.matmul(Kinv, F)


# seperate nodal displacements for each element.
u_elem = np.zeros([16, num_elem])
for el in range(num_elem):
    for num_nodes in range(8):
        indx = disp_map[connect_mat[el][num_nodes]]
        u_elem[num_nodes*2][el] = u[indx[0]][0]
        u_elem[num_nodes*2 + 1][el] = u[indx[1]][0]
    

# compute stress.

# Stress for element 1
stress_element1,r1 = e[0].calculateStress(u_elem[:,0])

# Stress for element 2
stress_element2,r2 = e[1].calculateStress(u_elem[:,1])
  