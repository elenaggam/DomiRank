import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import scipy
import scipy.sparse
import os
import time


def domirank(G, sigma = -1, dt = 0.1, epsilon = 1e-5, maxIter = 10000, checkStep = 10):
    '''
    G is the input graph as a (preferably) sparse array.
    This solves the dynamical equation presented in the Paper: "DomiRank Centrality: revealing structural fragility of
    complex networks via node dominance" and yields the following output: bool, DomiRankCentrality
    Here, sigma needs to be chosen a priori.
    dt determines the step size, usually, 0.1 is sufficiently fine for most networks (could cause issues for networks
    with an extremely high degree, but has never failed me!)
    maxIter is the depth that you are searching with in case you don't converge or diverge before that.
    Checkstep is the amount of steps that you go before checking if you have converged or diverged.
    
    
    This algorithm scales with O(m) where m is the links in your sparse array.
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        G = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
    else:
        G = G.copy()
    
    # if sigma == -1:
    #     sigma = optimal_sigma(G, dt=dt, epsilon=epsilon, maxIter = maxIter, checkstep = checkstep) 
    
    # maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float32)
    # j = 0
    pGAdj = sigma*G.astype(np.float32)
    Psi = np.zeros(pGAdj.shape[0]).astype(np.float32)
    dt = np.float32(dt)
    boundary = epsilon*pGAdj.shape[0]*dt
    # conv_iter = 0
    for i in range(maxIter):
        tempVal = ((pGAdj @ (1-Psi)) - Psi)*dt 
        Psi += tempVal.real
        if i% checkStep == 0:
            if np.abs(tempVal).sum() < boundary:
                # conv_iter = i
                break
            # maxVals[j] = tempVal.max()
            # if i == 0:
            #     initialChange = maxVals[j]
            # # si el valor máximo de tempVal es mayor que el valor máximo anterior, y el valor máximo anterior es mayor que el valor máximo anterior a ese, entonces estamos divergiendo
            # if j > 0:
            #     if maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
            #         conv_iter = i+1
            #         return False, Psi, conv_iter
            # j+=1
    # if conv_iter == 0:
    #     conv_iter = maxIter
    return Psi/np.max(Psi) #normalize to [0,1]

def remove_node(G, removedNode):
    '''
    removes the node from the graph by removing it from a networkx.Graph type, or zeroing the edges in array form.
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if type(removedNode) == int:
            G.remove_node(removedNode)
        else:
            for node in removedNode:
                G.remove_node(node) #remove node in graph form
        return G
    elif type(G) == scipy.sparse._arrays.csr_array:
        # .eye(m, n=none, k=0): Returns a sparse matrix (m x n) where the kth diagonal is all ones and everything else is zeros.
        diag = scipy.sparse.csr_array(scipy.sparse.eye(G.shape[0])) 
        diag[removedNode, removedNode] = 0 #set the rows and columns that are equal to zero in the sparse array
        G = diag @ G 
        return G @ diag
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')

def get_component_size(G, strong = False):
    '''
    here we get the largest component of a graph, either from scipy.sparse or from networkX.Graph datatype.
    1. The argument changes whether or not you want to find the strong or weak - connected components of the graph'''
    
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key = len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key = len)
        else:
            GMask = max(nx.connected_components(G), key = len)
        G = G.subgraph(GMask)
        return len(GMask)     
       
    elif type(G) == scipy.sparse._arrays.csr_array:
        if strong == False:
            connection_type = 'weak'
        else:
            connection_type = 'strong'
        _, lenComponent = scipy.sparse.csgraph.connected_components(G, directed = True, connection = connection_type, return_labels = True)
        return np.bincount(lenComponent).max() # get the size of the largest component by counting the number of nodes in each component and returning the maximum count
    else:
        raise TypeError('You must input a networkx.Graph Data-Type or scipy.sparse.csr array')


