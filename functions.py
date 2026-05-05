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

def relabel_nodes(G, yield_map = False):
    '''relabels the nodes to be from 0, ... len(G).
    1. Yield_map returns an extra output as a dict. in case you want to save the hash-map to retrieve node-id'''
    if yield_map == True:
        nodes = dict(zip(range(len(G)), G.nodes()))
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G, nodes
    else:
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G

# attack functions

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
    elif type(G) == scipy.sparse.csr_array:
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
       
    elif type(G) == scipy.sparse.csr_array:
        if strong == False:
            connection_type = 'weak'
        else:
            connection_type = 'strong'
        _, lenComponent = scipy.sparse.csgraph.connected_components(G, directed = True, connection = connection_type, return_labels = True)
        return np.bincount(lenComponent).max() # get the size of the largest component by counting the number of nodes in each component and returning the maximum count
    else:
        raise TypeError('You must input a networkx.Graph Data-Type or scipy.sparse.csr array')

def get_largest_component(G, strong = False):
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key = len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key = len)
        else:
            GMask = max(nx.connected_components(G), key = len)
        G = G.subgraph(GMask)
        return G     
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')

def get_link_size(G):
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        links = G.number_of_edges() #convert to scipy sparse if it is a graph 
    elif type(G) == scipy.sparse.csr_array:
        links = G.sum() # caso no dirigido no sería/2?
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return links

def generate_attack(centrality, node_map = False):
    '''we generate an attack based on a centrality measure - 
    you can possibly input the node_map to convert the attack to have the correct nodeID'''
    if node_map == False:
        node_map = range(len(centrality)) 
    else:
        node_map = list(node_map.values())
    # dictionary: key=Node, value=centrality
    zipped = dict(zip(node_map, centrality)) 
    # list: attack strategy = nodes sorted by centrality (descending)
    attackStrategy = sorted(zipped, reverse = True, key = zipped.get)
    return attackStrategy

def network_attack_sampled(G, attackStrategy, sampling = 0):
    '''Attack a network in a sampled manner... recompute links and largest component after every xth node removal, according to some - 
    G: is the input graph, preferably as a sparse array.
    inputed attack strategy
    Note: if sampling is not set, it defaults to sampling every 1%, otherwise, sampling is an integer
    that is equal to the number of nodes you want to skip every time you sample. 
    So for example sampling = int(len(G)/100) would sample every 1% of the nodes removed'''
    
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
    else:
        GAdj = G.copy()

    N = GAdj.shape[0]
    initialComponent = get_component_size(GAdj) # for normalization to lcc(0) = 1
    initialLinks = get_link_size(G)

    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(N/100)

    # evolution of the links and lcc, according to sampling
    links = np.zeros(int(N/sampling)) 
    component = np.zeros(int(N/sampling))

    j = 0 # save data every N/sampling steps
    for i in range(N-1):
        if i%sampling == 0:
            if i != 0: # avoid saving the initial condition twice
                GAdj = remove_node(GAdj, attackStrategy[i-sampling:i]) # as we skipped sampling nodes, we remove the skipped nodes all at once
            links[j] = get_link_size(GAdj)/initialLinks # get the interest parameters (normalized)
            component[j] = get_component_size(GAdj)/initialComponent
            j += 1

    return component, links

def network_attack_plotting(G, attackStrategy, plotting, psi, directory = "Plots/"):
    '''Attack a network in a sampled manner... recompute links and largest component after every xth node removal, according to some - 
    G: is the input graph, preferably as a sparse array.
    inputed attack strategy
    Note: if sampling is not set, it defaults to sampling every 1%, otherwise, sampling is an integer
    that is equal to the number of nodes you want to skip every time you sample. 
    So for example sampling = int(len(G)/100) would sample every 1% of the nodes removed'''
    
    GAdj = G.copy()
    nx.set_node_attributes(GAdj, dict(enumerate(psi)), 'centr') # set the domirank as a node attribute for plotting purposes

    N = G.number_of_nodes()
    initialComponent = get_component_size(GAdj) # for normalization to lcc(0) = 1
    initialLinks = get_link_size(G)

    plotting_step = []
    for i in range(len(plotting)):
        plotting_step.append(int(N * plotting[i])) # convert percentage to number of removed nodes
    plotting_step = sorted(plotting_step) # sort the plotting steps in case they are not sorted
    if 0 not in plotting_step:
        plotting_step = [0] + plotting_step # add the initial condition to the plotting steps if it is not already there

    pos = nx.spring_layout(G, iterations=500, seed=42) # coherent layout positions for every plot

    for j in range(len(plotting_step)):
        if j != 0:
            GAdj = remove_node(GAdj, attackStrategy[plotting_step[j-1]:plotting_step[j]]) # as we skipped sampling nodes, we remove the skipped nodes all at once
            for k in attackStrategy[plotting_step[j-1]:plotting_step[j]]:
                if k in pos:
                    pos.pop(k) # remove the position of the removed nodes to avoid plotting them, but keeping original position

        links = get_link_size(GAdj)/initialLinks # get the interest parameters (normalized)
        component = get_component_size(GAdj)/initialComponent
        
        out = directory + f"p{plotting[j]:.2f}.png"
        
        psi2 = list(nx.get_node_attributes(GAdj, "centr").values()) # color scale
        nx.draw(GAdj, pos, cmap=plt.get_cmap('cividis'), node_color=psi2, font_color='white')    
        plt.title(f"p={plotting[j]:.2f},  lcc={component:.2f},  links={links:.2f}", fontsize = 12)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
    return
        
