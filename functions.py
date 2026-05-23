import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
from scipy.sparse.linalg import eigsh

####### paper #######

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
    # .astype(np.float32)
    # dt = np.float32(dt)
    pGAdj = sigma*G
    Psi = np.zeros(pGAdj.shape[0])
    boundary = epsilon*pGAdj.shape[0]*dt
    # conv_iter = 0
    for i in range(maxIter):
        tempVal = ((pGAdj @ (1-Psi)) - Psi)*dt 
        Psi += tempVal.real
        if i% checkStep == 0:
            if np.abs(tempVal).sum() < boundary:
                # print(f"Converged at iteration {i}")
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
    return Psi

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

def find_eigenvalue(G, minVal = 0, maxVal = 1, maxDepth = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10):
    '''
    G: is the input graph as a sparse array.
    Finds the largest negative eigenvalue of an adjacency matrix using the DomiRank algorithm.
    Currently this function is only single-threaded, as the bisection algorithm only allows for single-threaded
    exection. Note, that this algorithm is slightly different, as it uses the fact that DomiRank diverges
    at values larger than -1/lambN to its benefit, and thus, it is not exactly bisection theorem. I haven't
    tested in order to see which exact value is the fastest for execution, but that will be done soon!
    Some notes:
    Increase maxDepth for increased accuracy.
    Increase maxIter if DomiRank doesn't start diverging within 100 iterations -- i.e. increase at the expense of 
    increased computational cost if you want potential increased accuracy.
    Decrease checkstep for increased error-finding for the values of sigma that are too large, but higher compcost
    if you are frequently less than the value (but negligible compcost).
    '''
    x = (minVal + maxVal)/G.sum(axis=-1).max()
    minValStored = 0
    for i in range(maxDepth):
        if maxVal - minVal < epsilon:
            break
        if domirank(G, x, dt, epsilon, maxIter, checkStep)[0]:
            minVal = x
            x = (minVal + maxVal)/2
            minValStored = minVal
        else:
            maxVal = (x + maxVal)/2
            x = (minVal + maxVal)/2
        # if minVal == 0:
        #     print(f'Current Interval : [-inf, -{1/maxVal}]')
        # else:
        #     print(f'Current Interval : [-{1/minVal}, -{1/maxVal}]')
    finalVal = (maxVal + minVal)/2
    return -1/finalVal

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
    initialComponent = float(get_component_size(GAdj)) # for normalization to lcc(0) = 1
    initialLinks = float(get_link_size(G))

    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(N/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1
    
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

# optimal sigma

def process_iteration(q, i, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling):
    domiDist = domirank(spArray, sigma, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
    domiAttack = generate_attack(domiDist)
    ourTempAttack, __ = network_attack_sampled(spArray, domiAttack, sampling = sampling)
    finalErrors = ourTempAttack.sum()
    q.put(finalErrors) # save result in mp queue

def optimal_sigma(spArray, endVal = 0, startval = 0.000001, iterationNo = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10, maxDepth = 100, sampling = 0):
    ''' This part finds the optimal sigma by searching the space, here are the novel parameters:
    spArray: is the input sparse array/matrix for the network.
    startVal: is the starting value of the space that you want to search.
    endVal: is the ending value of the space that you want to search (normally it should be the eigenvalue)
    iterationNo: the number of partitions of the space between lambN that you set
    
    return : the function returns the value of sigma - the numerator of the fraction of (\sigma)/(-1*lambN)
    '''
    if endVal == 0:
        endVal = find_eigenvalue(spArray, maxDepth = maxDepth, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
    import multiprocessing as mp
    endval = -0.9999/endVal
    # array[start, end, step] making sure to include endval with + (endval-startval)/iterationNo
    tempRange = np.arange(startval, endval + (endval-startval)/iterationNo, (endval-startval)/iterationNo)
    processes = [] # storing parallel processes
    q = mp.Queue() # mp queue to store the results of each process
    for i, sigma in enumerate(tempRange):
        p = mp.Process(target=process_iteration, args=(q, i, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling))
        p.start() # start the process 
        processes.append(p) # save process info

    results = []
    for p in processes:
        p.join() # wait for the process to finish
        result = q.get() # get process result
        results.append(result)
    finalErrors = np.array(results)
    minEig = np.where(finalErrors == finalErrors.min())[0][-1] # index of lowest lcc curve area
    minEig = tempRange[minEig] # sigma of lowest lcc curve area
    return minEig, finalErrors # sigma and areas

####### end of paper #######

def old_optimal_sigma(spArray, endVal = 0, startval = 0.000001, iterationNo = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10, maxDepth = 100, sampling = 0, directory = None):
    '''optimal sigma sequentally (not parallelized)'''
    
    if endVal == 0:
        endVal = find_eigenvalue(spArray, maxDepth = maxDepth, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)

    endval = -0.9999/endVal # -0.9999/lambda
    # array[start, end, step] making sure to include endval with + (endval-startval)/iterationNo
    tempRange = np.arange(startval, endval + (endval-startval)/iterationNo, (endval-startval)/iterationNo)

    finalErrors = []

    for sigma in tempRange:
        Psi = domirank(spArray, sigma = sigma, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
        attack = generate_attack(Psi)
        lcc, _ = network_attack_sampled(spArray, attack, sampling = sampling) # get the lcc after attacking with the generated attack strategy
        finalErrors.append(lcc.sum()) # get the area under the lcc curve as a measure of the attack's effectiveness

    finalErrors = np.array(finalErrors)
    minEig = np.where(finalErrors == finalErrors.min())[0][-1] # index of lowest lcc curve area
    minEig = tempRange[minEig]

    if directory is not None:
        plt.plot(-tempRange*endVal, finalErrors)
        plt.xlabel("sigma/λ")
        plt.ylabel("Area under LCC curve")
        plt.savefig(directory + "sigma.png", dpi=300, bbox_inches='tight')
        plt.close()

    return minEig, finalErrors # sigma and areas

def old2_optimal_sigma(G, delta_sigma = 0.001, sampling = 0, dt = 0.1, epsilon = 1e-5, maxIter = 10000, checkStep = 10):
    
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G).astype(float) #convert to scipy sparse if it is a graph 
    else:
        GAdj = G.copy()
    
    eig = eigsh(GAdj, return_eigenvectors=False) # get the largest eigenvalue of the adjacency matrix
    sigma_max = -1.0/np.min(eig)
    sigma_range = np.arange(0.001, sigma_max, delta_sigma) 

    lcc_list = []

    for sigma in sigma_range:
        Psi = domirank(GAdj, sigma = sigma, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
        attack = generate_attack(Psi)
        lcc, _ = network_attack_sampled(GAdj, attack, sampling = sampling) # get the lcc after attacking with the generated attack strategy
        area_lcc = np.sum(lcc)
        lcc_list.append(area_lcc)
    
    optimal_sigma = sigma_range[np.argmin(lcc_list)]
    
    return optimal_sigma, lcc_list


def network_attack_plotting_step(G, attackStrategy, p, psi, title, ax):
    
    GAdj = G.copy()
    nx.set_node_attributes(GAdj, dict(enumerate(psi)), 'centr') # set the domirank as a node attribute for plotting purposes

    N = G.number_of_nodes()
    initialComponent = get_component_size(GAdj) # for normalization to lcc(0) = 1
    initialLinks = get_link_size(G)

    plotting_step = int(N * p)
    
    pos = nx.spring_layout(G, iterations=500, seed=200) # coherent layout positions for every plot

    GAdj = remove_node(GAdj, attackStrategy[0:plotting_step]) # as we skipped sampling nodes, we remove the skipped nodes all at once
    for k in attackStrategy[0:plotting_step]:
        if k in pos:
            pos.pop(k) # remove the position of the removed nodes to avoid plotting them, but keeping original position

    links = get_link_size(GAdj)/initialLinks # get the interest parameters (normalized)
    component = get_component_size(GAdj)/initialComponent
    

    psi2 = list(nx.get_node_attributes(GAdj, "centr").values()) # color scale
    
    nx.draw(GAdj, pos, cmap=plt.get_cmap('cividis'), node_color=psi2, font_color='white', ax=ax)    
    ax.set_title(title, fontsize = 20)

    return links, component

def network_attack_plotting(G, attackStrategy, p_values, centrality, titles, directory=None):
    '''This function plots the attack for different values of p, and saves the plots in the specified directory.'''
    
    if len(titles) != centrality.shape[0] or centrality.shape[0] != attackStrategy.shape[0]:
        print("Error: titles, centrality and attackStrategy must have the same length")
        return
    
    for i, p in enumerate(p_values):
        fig, axes = plt.subplots(2, 2, figsize=(12,12))
        ax = axes.flatten()
        for j in range(centrality.shape[0]):
            links, component = network_attack_plotting_step(G, attackStrategy[j], p, centrality[j], titles[j], ax[j])
        
        if directory is not None:
            out = directory + f"p_{int(p*100)}.png"
        else:
            out = f"p_{int(p*100)}.png"
        fig.suptitle(f"p = {p:.2f}", fontsize=22)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

    return


def average_domi_attack(N, network_function, network_args, base = "results/", avgN = 20, startval = 0.000001, iterationNo = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10, maxDepth = 100, sampling = 0):
    
    lcc_values = np.zeros(int(N/sampling)) # initialize lcc values for averaging
    links_values = np.zeros(int(N/sampling)) # initialize links values for averaging
    
    file = open(f"{base}sigma.txt", "w")
    avg_eig = 0
    avg_sigma = 0

    for j in range(avgN):
        G = network_function(**network_args)
        G = relabel_nodes(G) 
        sparse_G = nx.to_scipy_sparse_array(G)
        eigenvalues, _ = eigsh(nx.to_scipy_sparse_array(G).astype(float), k=1,which='SA')
        eig = np.min(eigenvalues)

        sigma, _ = old_optimal_sigma(sparse_G, endVal = eig, sampling = sampling, iterationNo=iterationNo, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep, maxDepth = maxDepth)
        psi = domirank(sparse_G, sigma, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
        attack = generate_attack(psi)
        lcc, links = network_attack_sampled(G, attack, sampling = sampling)

        file.write(f"{sigma:.4f}\t{eig:.4f}\n")
        avg_sigma += sigma
        avg_eig += eig
        lcc_values += np.array(lcc)
        links_values += np.array(links)

        del G, sparse_G, psi, lcc, links # free memory

    file.write(f"\n{avg_sigma/avgN:.4f}\t{avg_eig/avgN:.4f}\n")
    file.close()

    lcc_values /= float(avgN)
    links_values /= float(avgN)

    np.savetxt(f"{base}Domirank_averaged_lcc_links.txt", np.array([lcc_values, links_values]).T, fmt = "%.4f") # save the lcc and links values for the attack
    
    return

def average_attack(N, network_function, network_args, centrality_name, centrality_func, base = "results/", avgN = 20, sampling = 0):
    
    lcc_values = np.zeros(int(N/sampling)) # initialize lcc values for averaging
    links_values = np.zeros(int(N/sampling)) # initialize links values for averaging

    for j in range(avgN):
        G = network_function(**network_args)
        G = relabel_nodes(G) 

        centrality = centrality_func(G)
        if isinstance(centrality, dict):
            centrality = np.array(list(centrality.keys()))
        
        lcc, links = network_attack_sampled(G, centrality, sampling = sampling)

        lcc_values += np.array(lcc)
        links_values += np.array(links)

    lcc_values /= float(avgN)
    links_values /= float(avgN)

    np.savetxt(f"{base}{centrality_name}_averaged_lcc_links.txt", np.array([lcc_values, links_values]).T, fmt = "%.4f") # save the lcc and links values for the attack
    
    return


def changing_attack(GAdj, attackStrategy=[], centrality_func = None, node_map = {}, sampling = 0, i = 0):
    if centrality_func is not None:
        centr = centrality_func(GAdj)
        if isinstance(centr, dict):
            psi = list(centr.values())
        else:
            psi = centr
        attackStrategy = generate_attack(psi, node_map=node_map) # update attack strategy based on new centrality
        attacked = attackStrategy[:sampling]
        node_map = {k: v for k, v in node_map.items()
                    if v not in attacked}
        node_map = relabel_dict(node_map) # relabel the node map keys to be from 0,...,len(node_map)-1 for the next iteration
        
    elif len(attackStrategy) > 0:
        attacked = attackStrategy[i-sampling:i] 
    else:
        raise ValueError("Attack strategy is empty, please provide an attack strategy or a centrality function to generate one.")
    return attacked, node_map

def choose_recovered_node(to_recover, method, p):
    '''
    chooses the node to recover based on the method, either random or sequential.
    '''
    check = 0
    if method == "random":
        chosen = np.random.choice(to_recover)
    elif method == "sequential":
        chosen = to_recover[0]
    if np.random.rand() < p: # with probability p, we recover the node
        check = 1
    return chosen, check

def recovery_step(to_recover, GAdj, method, p, G_original):
    '''
    node recovery step
    '''
    chosen, check = choose_recovered_node(to_recover, method, p)
    if check == 1: # if we recover the node, we add it to the
        GAdj.add_node(chosen, **G_original.nodes[chosen]) # add the node to the recovering graph
        to_recover.remove(chosen)

        # restore chosen's edges (if the neighbors have been recovered)
        for neighbor in G_original.neighbors(chosen):
            if GAdj.has_node(neighbor):
                edge_data = G_original.get_edge_data(chosen, neighbor)
                GAdj.add_edge(chosen, neighbor, **edge_data)

    return check

def network_recovery(to_recover, G_original, GAdj, p, sampling = 0, links = [], component = [], method = "random"):
    '''
    given the list of nodes from G_original to recover, the current graph GAdj,
    we recover the nodes with probability p according to method, 
    and save the links and component size after every recovery step, until we have recovered all the nodes.
    '''
    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(len(to_recover)/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1

    initialComponent = float(get_component_size(G_original)) # for normalization to lcc(0) = 1
    initialLinks = float(get_link_size(G_original))

    while len(to_recover) > 0: # we want to recover all the nodes
        
        for s in range(sampling):
            if len(to_recover) == 0: # if there are no nodes to recover, we skip the recovery process
                break
            check = recovery_step(to_recover, GAdj, method, p, G_original) 
        
        if check == 1: # if we recovered a node, we save the links and component size
            links.append(get_link_size(GAdj)/initialLinks) # get the interest parameters (normalized)
            component.append(get_component_size(GAdj)/initialComponent)
        else:
            links.append(links[-1]) 
            component.append(component[-1])

    return component, links

def network_attack_recovery(G_original, p, attackStrategy, method = "random", sampling=0, centrality_func = None):
    ''' 
    attack and recover network
    when not using domirank, centrality_func should be used if we want to update the attack strategy
    '''

    if type(G_original) != nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.from_scipy_sparse_array(G_original) #convert to scipy sparse if it is a graph 
    else:
        GAdj = G_original.copy()

    N = nx.number_of_nodes(GAdj)
    initialComponent = float(get_component_size(GAdj)) # for normalization to lcc(0) = 1
    initialLinks = float(get_link_size(GAdj))

    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(N/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1
    
    # evolution of the links and lcc, according to sampling
    links = []
    component = []

    to_recover = []
    node_map = dict(zip(range(N), GAdj.nodes())) # map from 0,...,N-1 to the original node IDs

    for i in range(N-1):
        if i%sampling == 0:
            if i != 0: 
                # attack process
                attacked, node_map = changing_attack(GAdj, attackStrategy=attackStrategy, centrality_func=centrality_func, node_map=node_map, sampling=sampling, i=i) 
                to_recover.extend(attacked) 
                GAdj = remove_node(GAdj, attacked) 

                for s in range(sampling): # in each step we remove sampling nodes, thus we take sampling time steps
                    # recovery process
                    if len(to_recover) == 0: # if there are no nodes to recover, we skip the recovery process
                        break

                    check = recovery_step(to_recover, GAdj, method, p, G_original)
                # we save the links and lcc independently of a node being recovered, bc we're still attacking the network
                links.append(get_link_size(GAdj)/initialLinks) # get the interest parameters (normalized)
                component.append(get_component_size(GAdj)/initialComponent)


    component, links = network_recovery(to_recover, G_original, GAdj, p, sampling = sampling, links = links, component = component, method = method) # recover the remaining nodes after the attack process is finished

    return component, links

def collective_influence(G, l = 2):
    '''Collective influence centrality measure'''
    
    CI = {}
    for i in G.nodes():
        neighbors = nx.single_source_shortest_path_length(G, i, cutoff=l).keys() # get the neighbors within l distance
        CI[i] = (G.degree[i]-1) * sum(G.degree[j]-1 for j in neighbors if j != i) # calculate the collective influence centrality
    return CI

def relabel_dict(dict):
    '''relabels the keys of dict to be the values of dict, where dict is a mapping from old keys to new keys.'''
    new_key = 0
    new_dict = {}
    for key, value in dict.items():
        new_dict[new_key] = value
        new_key += 1
    
    return new_dict