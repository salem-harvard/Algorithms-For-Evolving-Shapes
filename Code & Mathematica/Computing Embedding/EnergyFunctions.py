# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:19:21 2018

@author: Salem

In this scripy I will define the energy of the shell and related methods. 

Methods: 
    
    energy(u, starting_verts, bar_lengths, spring_consts,  fixed_disps = np.zeros(0), fixed_indices=[]):
        After setting the fixed_verts, calculates the energy using the edge matrix.
    
    energy_Der(u, DynMat,  fixed_disps, fixed_indices=[]):    
        
"""
    
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.linalg as sla
import scipy.optimize as op

import MeshHandling as MH


import importlib
importlib.reload(MH)

from copy import deepcopy
#===========================================================================================================================================
# implements a change in prefered lengths and returns the new equlibrium position
#===========================================================================================================================================
def update_mesh(init_verts, edges, sqr_bar_lengths=np.zeros(0), sqr_length_change=np.zeros(0), 
                spring_consts=np.zeros(0), insertIndices=(-1), deleteIndices=(-1)):
    '''
    Updates the edges of the mesh and returns the new positions
    
    input: 
        init_verts: Initial vertices
        edge_array: edge array
        steps: number of time steps 
        
    '''
    
    global num_of_verts, num_of_edges
    num_of_verts = init_verts.shape[0]
   
    edge_array = deepcopy(edges)
    #diherdralEdges = MH.DihedralVertices(num_of_verts, edge_array)
    
    #edge_array = np.vstack((edge_array, diherdralEdges))
    
    #thickness = 0.01
    
    
    
    num_of_edges = edge_array.shape[0]
    
    #connectivity dependent matrices that help in the calculation
    edge_mat1 = MH.makeEdgeMatrix1(edge_array)
    edge_mat2 = MH.makeEdgeMatrix2(edge_array)
    
    # get the adjacency and neighbor and edge map matrices. 
    adjMat, edgeMap = MH.makeAdjacencyMatrix(edge_array)
     
    
    #if initializing the square lengths to be those of the intial mesh
    if(sqr_bar_lengths.size==0):
        sqr_init_lengths = np.sum((init_verts[edge_array[:,0]] - init_verts[edge_array[:,1]])**2, axis=1)
        sqr_bar_lengths = sqr_init_lengths
    
    #add the bar lengths of the dihedral vertices, just the current length    
    #sqr_init_lengths = np.sum((init_verts[edge_array[num_of_edges//2:,0]] - init_verts[edge_array[num_of_edges//2:,1]])**2, axis=1)
    #sqr_bar_lengths = np.hstack((sqr_bar_lengths, sqr_init_lengths))
    
    
    if(sqr_length_change.size==0):
        sqr_length_change =  sqr_bar_lengths*0.01
        
    #sqr_length_change = np.vstack((sqr_length_change, np.zeros_like(sqr_length_change)))
    
    
    if(spring_consts.size == 0):
        spring_consts = np.ones(num_of_edges)
        
    #spring_consts = np.hstack((spring_consts, np.ones(num_of_edges//2)))
    
    
    new_verts = deepcopy(init_verts)
    
    if(insertIndices[0] < 0):
        insertIndices = (0,0,0)
       
    #print(spring_consts.shape, sqr_length_change.shape, sqr_bar_lengths.shape)
        
    displacement = get_displacement(new_verts, edge_array, sqr_bar_lengths, spring_consts, sqr_length_change,
                       edge_mat1=edge_mat1, edge_mat2=edge_mat2, adjMat=adjMat, edgeMap=edgeMap, deleteIndices=deleteIndices)  
        
    new_verts += np.insert(displacement, insertIndices, 0).reshape((num_of_verts, 2))
        
        
    return new_verts   
#===========================================================================================================================================


#===========================================================================================================================================
# Finds the approximate new equilibrium positions by inverting the dynamical matrix 
#===========================================================================================================================================
def get_displacement(vertices, edge_array, sqrdBarLengths, spring_consts, sqr_length_change,
                       edge_mat1=np.zeros(0), edge_mat2=np.zeros(0), adjMat=np.zeros(0), edgeMap=np.zeros(0), deleteIndices=np.zeros(0)):
    '''
    For an energy E(x, l) where x is the dynamical variable and l are the parameters we can find the change in the minizer x_min
    when we change the paramters l. This method implements this and returns the change in the minimizer. 
    
    input: 
        rigidityMat: rigidity matrix
        dynMat: dynamical matrix
        sqr_length_change: the change in the length square of the bars
        spring_consts: spring constants
    '''
    
    deff, M =  makeModifiedDynMat(vertices, edge_array, sqrdBarLengths, spring_consts,
                                  sqr_length_change, edge_mat1=edge_mat1, edge_mat2=edge_mat2, adjMat=adjMat, edgeMap=edgeMap)
    
    
    # delete the rows and columns that correspond to the axes that affect 
    if(deleteIndices[0] < 0):
        deleteIndices = (0,1,2)
        
    deff = np.delete(deff, deleteIndices, axis=0)
    deff = np.delete(deff, deleteIndices, axis=1)
    
    M = np.delete(M, deleteIndices)
    
   
    #res = op.minimize(energy, np.zeros(M.size, ), method='Newton-CG', args=(deff, M), 
                     # jac=energy_grad, hess=energy_hess, options={'xtol': 1e-8, 'disp': False})
    
    return np.dot(sla.pinv(deff), M)
   
    
#===========================================================================================================================================
    
#===========================================================================================================================================
# Finds the approximate new equilibrium positions by inverting the dynamical matrix 
#===========================================================================================================================================
def makeModifiedDynMat(vertices, edge_array, sqrdBarLengths, spring_consts, sqr_length_change,
                       edge_mat1=np.zeros(0), edge_mat2=np.zeros(0), adjMat=np.zeros(0), edgeMap=np.zeros(0)):
    '''
    the modified dynamical matrix that results from taking the second derivative of the energy. 
    this is different from usual expression because of the absense of an embedding of the edges. 
    
    input: 
        vertices: the current position of the vertices. 
        edge_array: the edge list
        edge_mat1: first edge matrix (defined in MeshHandling)
        edge_mat2: second edge matrix (defined in MeshHandling)
        sprin_const: spring constants. 
        adjMat: adjacency matrix.
        neibList: for each vertex gives the list of it's neighors.
        edgeMap: num_vertices x num_vertices array that contains the index of the edge connecting the vertices [i,j]
    '''
    
    num_dims = 2 #number of dimensions
    
    if(edge_mat1.size==0):
        #connectivity dependent matrices that help in the calculation
        edge_mat1 = MH.makeEdgeMatrix1(edge_array)
        edge_mat2 = MH.makeEdgeMatrix2(edge_array)
    
        # get the adjacency and neighbor and edge map matrices. 
        adjMat, edgeMap = MH.makeAdjacencyMatrix(edge_array, numOfVerts=num_of_verts)
        
        
    rigidityMatrix = MH.makeRigidityMat(vertices.flatten(), edgeMat1=edge_mat1, edgeMat2=edge_mat2, 
                                        numOfVerts=num_of_verts, numOfEdges=num_of_edges)
    
    dynMat = MH.makeDynamicalMat(RigidityMat=rigidityMatrix, springK=spring_consts,numOfEdges=num_of_edges)
    
    squared_lengths = np.sum((vertices[edge_array[:,0]] - vertices[edge_array[:,1]])**2, axis=1)
    
    d1 = -0.5*adjMat*((spring_consts*(squared_lengths - sqrdBarLengths))[edgeMap])
    
    #since d1 is symmetric we can sum over any axes
    d2 = -np.diag(np.sum(d1, axis=1))
    
    d = np.kron(d1 + d2, np.eye(num_dims))
    
    #M is the matrix to be multiplied by the inverse dynamical matrix
    return dynMat + d, 0.5* np.dot(np.transpose(rigidityMatrix), spring_consts*sqr_length_change)
   
    
#===========================================================================================================================================
