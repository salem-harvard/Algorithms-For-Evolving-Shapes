# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:26:34 2018

@author: Salem

This script includes methods for creating and handling of Meshes. 

Contains: 
    
    MakeCylVerts (maxTheta=10, maxZ=10, radius = 0.5, length = 2, cap_size):
        returns the vertices of a cylinder. Centered on the origin with two cap vertices on the ends. 
"""
import numpy as np

import matplotlib.pyplot as plt

#=========================================================================================================================
# defining the current working directory
#=========================================================================================================================
from pathlib import Path
import os
cwd = Path(os.getcwd())


results_dir = cwd / "Results" 
#=========================================================================================================================


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


#=========================================================================================================================
# make a simple hexagonal lattice
#=========================================================================================================================
def hex_lattice(width=20, height=20):
    '''
    creates a hexagonal lattice from a square lattice by shifting the odd numbered rows (counting starts from zero).
    
    Width: number of vertices in the x-direction.
    
    '''
    #assuming the square lattice has dimensions of 1 x 1, (points are in [0,1])
    x_spacing = 1/width
    y_spacing = 1/height
    
    vertices = np.array([[(i + 0.5* np.mod(j, 2)) * x_spacing , j * y_spacing]
                      for j in range(height) for i in range(width)])
    
    #shows the created lattice
    plt.scatter(vertices[:,0], vertices[:,1])
    plt.show()
    
    #horizontal edges
    edges = [[n + m * width, n + 1 + m * width] for m in range(height) for n in range(width - 1)]
    
    # up and to the right, starting from even rows
    edges.extend([[n + 2*m*width, n + width + 2*m*width] for m in range(height//2) for n in range(width)])
    
    # up and to the right, starting from odd rows
    edges.extend([[n + (2*m + 1)*width, n + width + 1 + (2*m + 1)*width] for m in range((height - 1)//2) for n in range(width - 1)])
    
    # down and to the right, starting from even rows
    edges.extend([[n + 2*m*width, n - width + 2*m*width] for m in range(1, (height + 1)//2) for n in range(width)])
    
    # down and to the right, starting from odd rows
    edges.extend([[n + (2*m + 1)*width, n - width + 1 + (2*m + 1)*width] for m in range((height)//2) for n in range(width - 1)])    
        
    return vertices, np.array(edges)
#=========================================================================================================================
#=========================================================================================================================  

#======================================================================================================================================
# returns the adjacency matrix as an array
#======================================================================================================================================
def makeAdjacencyMatrix(edgeArray, numOfVerts=-1):
    """
    makeAdjacencyMatrix(edgeList):
        Takes in the edgeArray then converts it to a list, which has elements of the form [vert1, vert2]
        and finds the (2 numOfVerts x 2 numOfVerts) adjacency matrix.
        
    
    """
    
    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))

    adjacencyMat = np.zeros((numOfVerts, numOfVerts), dtype=int) 
    edgeMap = np.zeros((numOfVerts, numOfVerts), dtype=np.int) 
    
    for eIndex, edge in enumerate(edgeArray):
        adjacencyMat[edge[1], edge[0]] = adjacencyMat[edge[0], edge[1]]= 1
        edgeMap[edge[1], edge[0]] = edgeMap[edge[0], edge[1]]= eIndex
        
    return adjacencyMat, edgeMap
#=====================================================================================================================================

#=====================================================================================================================================
# returns the Edge Matrix given the edge array
#=====================================================================================================================================
def makeEdgeMatrix1(edgeArray, numOfVerts=-1, numOfEdges=-1, useSpringK = False, springK = -np.ones(1)):
    """
    makeEdgeMatrix(edgeArray, numOfVerts=-1, numOfEdges=-1, useSpringK = False, springK = -np.ones(1)): 
        gives the edge matrix, which has dimenstions (3*numOfEdges, 3*numOfVerts).
        For each edge there are three rows in the matrix, the row is only nonzero at the positions 
        corresponding to the points connected by that edge, one of them will be 1 the other will be -1. 
        There will be a row for each x,y,z component
        When useSpringK is True, each edge will be multiplied by the spring constant which is a convenient thing
        
        Example: verts, edges = squareLattice(2)
            EdgeMat1 = makeEdgeMatrix3(edges); EdgeMat1
       Out:  array([[ 1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
       [ 1.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0., -1.],
       [ 1.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.]])
    """
    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))
       
    if numOfEdges < 0:
        numOfEdges = edgeArray.size//2
        
    if useSpringK:
        if (springK < 0).all():
            springK = np.ones(numOfEdges)
    
        
    edgeMat = np.zeros((2*numOfEdges, 2*numOfVerts))
    
    for edgeNum, edge in enumerate(edgeArray):
        if not useSpringK:
            edgeMat[2*edgeNum, 2*edge[0]] = 1 
            edgeMat[2*edgeNum + 1, 2*edge[0] + 1] = 1 
            
            edgeMat[2*edgeNum, 2*edge[1]] = -1 
            edgeMat[2*edgeNum + 1, 2*edge[1] + 1] = -1 
            
        else:
            edgeMat[2*edgeNum, 2*edge[0]] = 1 *springK[edgeNum]
            edgeMat[2*edgeNum + 1, 2*edge[0] + 1] = 1 *springK[edgeNum]
            
            edgeMat[2*edgeNum, 2*edge[1]] = -1 *springK[edgeNum]
            edgeMat[2*edgeNum + 1, 2*edge[1] + 1] = -1  *springK[edgeNum]
        
    return edgeMat
#=====================================================================================================================================
    
#=====================================================================================================================================
# returns the Edge Matrix given the edge array
#=====================================================================================================================================
def makeEdgeMatrix2(edgeArray, numOfVerts=-1, numOfEdges=-1):
    """
    makeEdgeMatrix2(edgeArray, numOfVerts=-1, numOfEdges=-1, useSpringK = False, springK = -np.ones(1)): 
        gives the edge matrix, which has dimenstions (numOfEdges, 3*numOfEdges).
        For each edge there is a row in the matrix, the row is only nonzero at 2 positions in which 
        it is equal to 1, this is used for adding together the two rows corresponding to the different
        x and y componenets that resulted from multiplying edgeMatrix1 with the vertices. 
        
        Example: verts, edges = squareLattice(2)
            EdgeMat2 = makeEdgeMatrix2(edges); EdgeMat2
            array([[ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.]])
    """

    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))
    if numOfEdges < 0:
        numOfEdges = edgeArray.size//2
        
    edgeMat = np.zeros((numOfEdges, 2*numOfEdges),dtype=np.dtype('int32')) 
    
    for edgeNum, edge in enumerate(edgeArray):
        edgeMat[edgeNum, 2*edgeNum] = 1 
        edgeMat[edgeNum , 2*edgeNum + 1] = 1 
       
        
    return edgeMat
#===================================================================================================================================
#===================================================================================================================================
# returns the Rigidity Matrix as an array
#===================================================================================================================================
def makeRigidityMat(verts, edgeArray=np.array([0]), numOfVerts=-1, numOfEdges=-1, edgeMat1 = np.zeros(0), edgeMat2 = np.zeros(0)):
    """
    makeRigidityMat(verts, edgeArray, numOfVerts=-1, numOfEdges=-1,method):
        Takes in the edgeArray then finds Rigidity matrix. The rigidity matrix helps
        to find the bond stretching to linear order in displacement u which has 
        size = 2 numOfVerts. Bond stretchings are equal to 
        dl_e = R_ei * u_i, where i is summed over.
        
        The method parameter desides how the rigidity matrix will be computed. When method = 1
        the edgeMatrices will be used, which is useful when the vertex positions are minimized over. 
        verts should be flattened when this method is used
        
    Example1: 
            sq = squareLattice(2, randomize=False); 
            edgeMat1= makeEdgeMatrix1(sq[1])
            edgeMat2 = makeEdgeMatrix2(sq[1])
            R = makeRigidityMat(sq[0].flatten(), edgeMat1=edgeMat1, edgeMat2=edgeMat2)
            R 
        Out: array([[ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  1.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  1.,  1.]])
    
    Example2:
        (verts, edges) = squareLattice(2, randomize=False); 
        edgeMat1 = 
            R = makeRigidityMat(verts, edges) ;R
      array([[ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  1.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  1.,  1.]])       
    """
    
    num_dimensions =2
    
    if not (edgeMat1.size == 0):
        
        RMat = np.dot(edgeMat1, verts)
        RMat = np.multiply(edgeMat1.transpose(), RMat).transpose()
        return np.dot(edgeMat2, RMat)
    
    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))
    if numOfEdges < 0:
        numOfEdges = edgeArray.size//2
      
    RigidityMat = np.zeros((numOfEdges, num_dimensions * numOfVerts))
    
    for edgeNum, edge in enumerate(edgeArray):
        t = np.zeros((numOfVerts, num_dimensions))
        t[edge[1]] = verts[edge[1]] - verts[edge[0]]
        t[edge[0]] = verts[edge[0]] - verts[edge[1]]
        RigidityMat[edgeNum] = t.flatten()
    
    return RigidityMat
#=====================================================================================================================================
    
#====================================================================================================================================
# returns the Rigidity Matrix as an array
#====================================================================================================================================
def makeDynamicalMat(edgeArray=np.zeros(1), verts=np.zeros(1), RigidityMat= np.zeros(1), springK= np.zeros(1),  
                     numOfVerts=-1, numOfEdges=-1, negativeK=False):
    """
    makeDynamicalMat(verts, edgeArray, numOfVerts=-1, numOfEdges=-1):
        Takes in the edgeArray then finds dynamical matrix. The dynamical matrix
        help in calculating the potential energy of a displacement u which has 
        size = 2 numOfVerts. The energy is given by E[u] = u.T D u.
        
    Example in 2D: 
            (verts, edges) = squareLattice(2, randomize=False); 
             makeDynamicalMat(edgeArray=edges, RigidityMat=R)
        Out: array([[ 2.,  1.,  0.,  0., -1.,  0., -1., -1.],
       [ 1.,  2.,  0., -1.,  0.,  0., -1., -1.],
       [ 0.,  0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
       [-1., -1., -1.,  0.,  0.,  0.,  2.,  1.],
       [-1., -1.,  0.,  0.,  0., -1.,  1.,  2.]])
    """
    if numOfEdges < 0:
            if(not edgeArray.any()):
                raise NameError("Please either provide the the number of edges or the edge array")
            numOfEdges = edgeArray.size//2
            
            
    if(not RigidityMat.any()):
        #print("This is not supposed to be true during minimization because we would be using a rigidity matrix")
        if not verts.any():
            raise NameError("Please either provide the rigidity matrix or the vertices for calculating the dynamical matrix")
        if numOfVerts < 1:
            numOfVerts = len(set(list(edgeArray.flatten())))

        RigidityMat = makeRigidityMat(verts, edgeArray, numOfVerts, numOfEdges) 
        #print(RigidityMat.shape)
        
    if(not springK.any()):
        springK = np.ones(numOfEdges)

    if not negativeK:
        dynMat = np.dot(np.dot(RigidityMat.transpose(), np.diag(springK**2)), RigidityMat)
    else:
        dynMat = np.dot(np.dot(RigidityMat.transpose(), np.diag(springK)), RigidityMat)
    return dynMat
#=====================================================================================================================================
    
#=================================================================================
#flatten a list 
#=================================================================================
def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis
#=================================================================================


#=================================================================================
#find the neighbor list from the edge list
#=================================================================================
def NeibsFromEdges(numVerts, EdgeList):
    '''
        Returns a Neighbor list and a map between the edges list and Neighbor list.
        This returns two python lists because the cap rows will be bigger. 
        Can be easily adjusted to exclude cap rows and be returned as an array
    '''

    #numEdges = EdgeList[:, 0].size
#for each row NeibList gives the neighbor indices of the vertex correspoding to the row index                       
    NeibList = [[]]*numVerts
#For each index in a row MapNL will point to the correct edge index               
    MapNL = [[]]*numVerts
               
    for Vindx in np.nditer(np.arange(numVerts)):
        for Eindx, edge in enumerate(EdgeList): 
            if edge[0] == Vindx:
                NeibList[Vindx] = [NeibList[Vindx], edge[1]]
                MapNL[Vindx] = [MapNL[Vindx], Eindx]
            elif edge[1] == Vindx:
                NeibList[Vindx] = [NeibList[Vindx], edge[0]]
                MapNL[Vindx] = [MapNL[Vindx], edge[0]]
                
            NeibList[Vindx] = flatten(NeibList[Vindx])
            MapNL[Vindx] = flatten(MapNL[Vindx])
                
    return (NeibList, MapNL)
#=================================================================================


#=================================================================================
#find the neighbor list from the edge list
#=================================================================================
def DihedralVertices(numVerts, EList):
    ''' 
        Calculates the dihedral vertices for the triangulation from edge list by first 
        finding the edge list.
        return (numEdges, 2) array containing the indices of the two 
        vertices corresponding to the triangles that include the edge
        Neibs can be an array or list
    '''
    Neibs = NeibsFromEdges(numVerts, EList)[0]
    
    numEdges = EList[:, 0].size
    
    FaceVerts = np.zeros((numEdges, 2), dtype=int)
    
    #loop over all the edges
    for i in range(numEdges):
        
    
        #find the two triangles intersecting at the edge by finding common neighbors
        FaceVerts[i] = np.intersect1d(Neibs[EList[i,0]], Neibs[EList[i,1]]) 
    
    return FaceVerts
#=================================================================================

import numpy.linalg as la
#=================================================================================
# normalize a list of vectors
#=================================================================================
def normalizeVecs(vecs, axis=1):
    '''
    normalizes a given array of vectors, each vector is laid out on the axis. 
    the default value of 1 normalizes each row.
    
    axis actually has to be 1
    '''
    norms = la.norm(vecs, axis=axis).reshape((np.size(vecs, 0), 1))
    
    #mask1 = norms != 0
    #mask2 = np.repeat(mask, np.size(vecs, 1), 1)
    #assuming axis = 1
    
    return vecs/norms
#=================================================================================    
    
#=================================================================================    
#find the angle between two arrays
#=================================================================================
def findAngle(v1, v2):
    '''
    find angle between two vectors
    '''
    v1_u = normalized(v1)
    v2_u = normalized(v2)
    return (np.arccos(np.clip(np.sum(v1_u * v2_u), -1.0, 1.0)))    
#=================================================================================    
    
    
#cotangent of an angle in rads
def cotan(angle):
    #cosine over sine
    return np.cos(angle)/np.sin(angle)

#return a normalized vector
def normalized(vector):
      return vector / la.norm(vector)    
  
# magnitude of each row
def RowNorms(x):
    return np.sum(np.abs(x)**2,axis=-1)**(1./2)
    
#square magnitude of each row
def RowSqrNorms(x):
    return np.sum(np.abs(x)**2,axis=-1)