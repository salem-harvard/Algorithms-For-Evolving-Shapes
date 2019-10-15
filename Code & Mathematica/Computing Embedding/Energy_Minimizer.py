# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 05:54:39 2018

@author: Salem
"""

import numpy as np
from numpy import linalg as la

import scipy.optimize as op

import matplotlib.pyplot as plt


#=========================================================================================================================
# find the embedding by minimizing the elastic energy
#=========================================================================================================================
def find_embedding():
    
    #initial vertices in coordinate space
    init_vertices, edges = hex_lattice()
    
    num_verts = init_vertices.shape[0]
    #num_edges = edges.shape[0]
    
    target_lengths_squared = get_target_lengths1(init_vertices, edges)
    
    edge_matrix1, edge_matrix2, edge_matrix3 = make_edge_matrix(edges, num_verts)
    
    res = op.minimize(elastic_energy, init_vertices.flatten(), args=(edge_matrix1, edge_matrix2, edge_matrix3,
                             target_lengths_squared), jac=energy_gradient, method='Newton-CG')
    
    new_verts = res.x.reshape((num_verts,2))
    
    plt.scatter(new_verts[:,0], new_verts[:,1])
    plt.show()
    
    print("total energy = ", res.fun)
    return res.fun

#=========================================================================================================================
# make a simple hexagonal lattice
#=========================================================================================================================
def hex_lattice(width=20, height=20):
    '''
    
    
    
    '''
    x_spacing = 1/width
    y_spacing = 1/height
    
    vertices = np.array([[(i + 0.5* np.mod(j, 2)) * x_spacing , j * y_spacing]
                      for j in range(height) for i in range(width)])
    
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
 
    
#=========================================================================================================================
# evaluate the elastic energy
#=========================================================================================================================

def elastic_energy(vertices, edge_matrix1, edge_matrix2, edge_matrix3, target_lengths_squared):
    
    return 0.25*np.sum((np.dot(edge_matrix2, np.dot(edge_matrix1, vertices)**2) - target_lengths_squared)**2)
#=========================================================================================================================
#=========================================================================================================================


#=========================================================================================================================
# make a simple hexagonal lattice
#=========================================================================================================================
def energy_gradient(vertices, edge_matrix1, edge_matrix2, edge_matrix3, target_lengths_squared):
    
    return np.dot(np.transpose(edge_matrix1), 
                  np.dot(np.dot(edge_matrix2, np.dot(edge_matrix1, vertices)**2) - target_lengths_squared, edge_matrix2) *  np.dot(
            edge_matrix1, vertices))
#=========================================================================================================================
#=========================================================================================================================    


#=========================================================================================================================
# make a simple hexagonal lattice
#=========================================================================================================================
def get_target_lengths1(init_verts, edges):
    '''
    
    '''
    
    num_edges = edges.shape[0]
    
    target_lengths_squared = np.zeros(num_edges)
    
    for i, e in enumerate(edges):
        displacement = init_verts[e[1]] - init_verts[e[0]]
        
        target_lengths_squared[i] = np.dot(displacement,  displacement)
    
    return 3*target_lengths_squared
#=========================================================================================================================
#=========================================================================================================================                 


#=========================================================================================================================
# make a simple hexagonal lattice
#=========================================================================================================================
def get_target_lengths2(metric, init_verts, edges):
    '''
    metric is a symmetric positive definite matrix
    '''
    
    num_edges = edges.shape[0]
    
    target_lengths_squared = np.zeros(num_edges)
    
    for i, e in enumerate(edges):
        displacement = init_verts[e[1]] - init_verts[e[0]]
    
        average_metric = 0.5*(metric[e[1]] + metric[e[2]])
        
        target_lengths_squared[i] = np.dot(displacement, np.dot(average_metric, displacement))
    
    return target_lengths_squared
#=========================================================================================================================
#=========================================================================================================================                 
    
    
def make_edge_matrix(edges, num_verts):
    
    num_edges = edges.shape[0]
    
    edge_matrix1 = np.zeros((2*num_edges, 2*num_verts))
    
    edge_matrix2 = np.zeros((num_edges, 2*num_edges))
    
    edge_matrix3 = np.zeros((num_edges, 2*num_verts))
    
    for e, edge in enumerate(edges):
        edge_matrix1[2*e, 2 * edge[0]] = 1
        edge_matrix1[2*e, 2 * edge[1]] = -1
        
        edge_matrix1[2*e + 1, 2 * edge[0] + 1] = 1
        edge_matrix1[2*e + 1, 2 * edge[1] + 1] = -1
    
        edge_matrix2[e, 2 * e] = 1
        edge_matrix2[e, 2 * e + 1] = 1
        
        edge_matrix3[e, 2*edge[0]] = 1
        edge_matrix3[e, 2*edge[1]] = 1
        edge_matrix3[e, 2*edge[0] + 1] = 1
        edge_matrix3[e, 2*edge[1] + 1] = 1
        
        
    return edge_matrix1, edge_matrix2, edge_matrix3
    
    
    
#===========================================================================================================================================
# returns the adjacency matrix as an array
#===========================================================================================================================================  
def makeAdjacencyMatrix(edges, numOfVerts=-1):
    """
    makeAdjacencyMatrix(edges):
        Takes in the edgeArray then converts it to a list, which has elements of the form [vert1, vert2] 
        and finds the (numOfVerts x numOfVerts) adjacency matrix.
    """
    
    edgeList = edges.tolist()
    
    if numOfVerts < 1:
        numOfVerts = len(set(list(edges.flatten())))

    adjacencyMat = np.zeros((numOfVerts, numOfVerts));     
    
    for i in range(numOfVerts):
        for j in range(numOfVerts):
            adjacencyMat[i, j] = 1 if [i, j] in edgeList or [j, i] in edgeList else 0
            
    return adjacencyMat
#===========================================================================================================================================    
    
    
import numdifftools as nd
def test_grad():
     
    #initial vertices in coordinate space
    init_vertices, edges = hex_lattice(width=4, height=4)
    
    #init_vertices = np.array([[0,0],[0, 1], [1,1]])
    #edges = np.array([[0,1], [1,2]])
    
    num_verts = init_vertices.shape[0]
    #num_edges = edges.shape[0]
    
    target_lengths_squared = get_target_lengths1(init_vertices, edges)
    
    edge_matrix1, edge_matrix2, edge_matrix3 = make_edge_matrix(edges, num_verts)
    
    print(make_edge_matrix(edges, num_verts), "\n\n")
    
    
    verts = init_vertices.flatten()
    
    print(elastic_energy(verts, edge_matrix1, edge_matrix2, edge_matrix3, target_lengths_squared), "\n\n")
    print(energy_gradient(verts, edge_matrix1, edge_matrix2, edge_matrix3, target_lengths_squared), "\n\n")
    print(op.approx_fprime(verts, elastic_energy, 0.000001, edge_matrix1, edge_matrix2, edge_matrix3, target_lengths_squared), "\n\n")
    
    
    
    return 1
    
    
    
    
    
    
    