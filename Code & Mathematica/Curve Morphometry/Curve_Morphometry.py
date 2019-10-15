#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 06:37:17 2018

@author: smosleh
"""

import numpy as np

from numpy import cos, sin, exp, transpose#, dot

#import numpy.linalg as la
import numpy.random as npr


import scipy.optimize as op
from time import clock


import matplotlib.pyplot as plt

from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceMatrix

from pathlib import Path
import os

cwd = Path(os.getcwd())

import xlrd 




def get_data():
    '''
    gets data from the excel file and rescales the parameters according to the arc length
    '''
    
    excel_file = cwd / "egg_shape_parameters.xlsx"
    
    data_indices = np.array([2,3,4,5,43,44,45,46,95,96, 97, 98, 104, 105, 106, 107])
    #np.array([2,3,4,5,43,44,45,46,95,96, 97, 98, 104, 105, 106, 107]) +100
    
    work_book =  xlrd.open_workbook(excel_file) 
    
    sheet = work_book.sheet_by_index(0)
    
    num_of_params = 6
    params = np.zeros((data_indices.size, num_of_params))
    
    for i, index in enumerate(data_indices):
        length =  sheet.cell_value(index, 7)
        for j in range(6):
            params[i, j] = sheet.cell_value(index, 8 + j)*(length**j)
    
    return params
    
def run_optimization():
    '''
    '''
    
    params = get_data()
    
    
    num_samples = 16
    

#---------------------------------------------------------------------------------------------------------------------------------------------------    
    NUM_OF_VERTICES = 200
    
    
    distances = np.zeros((num_samples, num_samples))
    
    for i in range(num_samples):
        for j in range(i + 1, num_samples): 
            print("working on the pair", (i, j))
            distances[i, j] = np.abs(compare_curves(params[i], params[j], num_of_verts=NUM_OF_VERTICES))
            distances[j, i] = distances[i,j]
#---------------------------------------------------------------------------------------------------------------------------------------------------  
                
 
    
# Plot distance matrix and make phylogenetic tree
#---------------------------------------------------------------------------------------------------------------------------------------------------    
    plt.matshow(distances)
    plt.colorbar()
    plt.show
    
    distaceMat = [list(distances[i, :i+1]) for i in range(16)]
    
    distaceMatrix = DistanceMatrix(names=['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4', 'c1', 'c2', 'c3', 'c4', 'd1', 'd2', 'd3', 'd4'],
                                   matrix=distaceMat)
    
    constructor = DistanceTreeConstructor()
    
    tree_up = constructor.upgma(distaceMatrix)
    
    tree_nj = constructor.nj(distaceMatrix)
    
    Phylo.draw_ascii(tree_nj)
    
    Phylo.draw_ascii(tree_up)
    
    return distances
#=========================================================================================================================
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================================  
    

def compare_curves(params1, params2, num_of_verts=100):
    
    # prepare the two curves for comparison    
#---------------------------------------------------------------------------------------------------------------------------------------------------    
    
    total_length = 1
    
    # the arc-length spacing between points in the first curve
    deltaS1 = total_length/(num_of_verts - 1)
    
    #coordinates in the initial or first curve
    init_coordinates = np.linspace(0, total_length, num_of_verts)

    #to be used in the initialization of the optimization problem
    final_coordinate_guess = np.linspace(0, total_length, num_of_verts) 
    
    #final_coordinate_guess = np.sort(npr.rand(100))
    
#---------------------------------------------------------------------------------------------------------------------------------------------------    
       
    init_curvatures = np.array(list(map(lambda x: curvature_function(x, params1), init_coordinates)))
    
    init_torsions = np.array(list(map(torsion_function, init_coordinates)))
    
    
    coeffs = np.ones(7); coeffs[0] = 1; coeffs[1] = 1; coeffs[3:5] = 1; coeffs[6] = 0.01
    
    print("cost of initial guess:", cost_function(final_coordinate_guess, params2, init_curvatures, init_torsions, deltaS1, coeffs))
    
    
    
    cons = ({'type': 'eq', 'fun': lambda x:  x[0]},
             {'type': 'eq', 'fun': lambda x: x[num_of_verts - 1] - total_length})
      
    
    
   # t = clock() 

    
    res = op.minimize(cost_function, final_coordinate_guess, args=(params2, init_curvatures, init_torsions, deltaS1, coeffs), constraints=cons)
    
    
    
    print("cost after minimization:", res.fun)
    
    #t = clock() - t

    #print("times taken to run:", t)
    
    return res.fun
    
    final_coordinates = res.x
    
    np.savetxt(cwd / "init_coordinates.npy", init_coordinates)
    np.savetxt(cwd / "init_curvatures.npy", init_curvatures)
    np.savetxt(cwd / "final_coordinates.npy", final_coordinates)
    
    final_curvatures = np.array(list(map(lambda x: curvature_function(x, params2), final_coordinates)))
    
    np.savetxt(cwd / "final_curvatures.npy", final_curvatures)
    
    
    
    

def cost_function(final_coordinates, final_params, init_curvatures, init_torsions, deltaS1, coeffs = np.ones(7)):
    '''
    '''  
    
    final_curvatures = np.array(list(map(lambda x: curvature_function(x, final_params), final_coordinates)))
    
    final_torsions = np.array(list(map(torsion_function, final_coordinates)))
    
    #relative change in length
    deltaL = (final_coordinates[1:] - final_coordinates[:-1])/deltaS1  -  1
    
    #change in curvature
    deltaK = (final_curvatures - init_curvatures)
    # average the changes accross each length
    deltaK = 0.5*(deltaK[1:] + deltaK[:-1])
    
    
    #change in torsion
    deltaT = (final_torsions - init_torsions)
    # average the changes accross each length
    deltaT = 0.5*(deltaT[1:] + deltaT[:-1])
    
    #the derivative with respect to arc length 
    gradL = (deltaL[1:] - deltaL[:-1])/deltaS1
    
    gradK = (deltaK[1:] - deltaK[:-1])/deltaS1
    
    gradT = (deltaT[1:] - deltaT[:-1])/deltaS1
    
    
    #make sure final coordiantes are increasing
    coord_change = final_coordinates[1:] - final_coordinates[:-1]
    
    A, B, C, D, E, F, G = coeffs
    
    #return the cost
    return A * np.dot(deltaL, deltaL * deltaS1) + B * np.dot(deltaK, deltaK * deltaS1) +  C * np.dot(deltaT, deltaT * deltaS1) + D * np.dot(
            gradL, gradL * deltaS1) + E * np.dot(gradK, gradK * deltaS1) +  F * np.dot(gradT, gradT * deltaS1) - G*np.dot(
                    coord_change, coord_change * deltaS1)

def curve_function(s, params=0):
    '''
    gives the form of the curve as a function of the (arc length) parameter s. 
    '''
    
    return [cos(s), sin(s), 0]

def curvature_function(s, shape_params):
    '''
    gives the curvature of a curve as a function of the (arc length) parameter s. 
    '''
    
    return np.dot([1, s, s*s, s*s*s, s**4, s**5], shape_params)
    
def torsion_function(s):
    '''
    gives the torsion of a curve as a function of the (arc length) parameter s. 
    '''

    return 0


def con1(x):
    return x[0]


def con2(x):
    return x[-1] - 1

def analyze_results(final_coordinates, init_curvatures, init_torsions, deltaS1):
    '''
    '''  
    
    final_curvatures = np.array(list(map(lambda x: curvature_function(x, s0=0.2), final_coordinates)))
    
    final_torsions = np.array(list(map(curvature_function, final_coordinates)))
    
    #relative change in length
    deltaL = (final_coordinates[1:] - final_coordinates[:-1])/deltaS1  -  1
    
    #change in curvature
    deltaK = (final_curvatures - init_curvatures)
    # average the changes accross each length
    deltaK = 0.5*(deltaK[1:] + deltaK[:-1])
    
    
    #change in torsion
    deltaT = (final_torsions - init_torsions)
    # average the changes accross each length
    deltaT = 0.5*(deltaT[1:] + deltaT[:-1])
    
    #the derivative with respect to arc length 
    gradL = (deltaL[1:] - deltaL[:-1])/deltaS1
    
    gradK = (deltaK[1:] - deltaK[:-1])/deltaS1
    
    gradT = (deltaT[1:] - deltaT[:-1])/deltaS1
    

    
    #return the deformations squared
    return [deltaL * deltaL ,  deltaK * deltaK, deltaT * deltaT ,
            gradL* gradL , gradK * gradK, gradT* gradT]












































































