import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh


def domirank(G, sigma = -1, dt = 0.1, maxsteps = 10000, epsilon = 1e-6, method = "euler"):
    '''
    Numerical (recursive) solution of the DomiRank centrality of the nodes in the graph G. 
    From the methods section in "DomiRank Centrality: revealing structural fragility 
    of complex networks via node dominance"
    
    Parameters:
    ------------
    G: a networkx graph
    sigma: float, optional (default = -1).
        The parameter of the DomiRank centrality. If -1, it is set to the upper limit of the convergence interval, which is -1/min(eigenvalues(G))
    dt: float, optional (default = 0.1).
        The time step of the numerical solution
    maxsteps: float, optional (default = 1e3).
        The maximum number of steps of the numerical solution
    epsilon: float, optional (default = 1e-5).
        The convergence threshold of the numerical solution. The convergence condition is ||gamma - gamma_prev|| < epsilon*N*dt, where N is the number of nodes in the graph
    method: str, optional (default = "euler").
        The numerical method to use. It can be "euler" or "heun". Heun's method is more accurate but slower than Euler's method.

    Returns:
    ------------
    gamma: numpy array of shape (N,) 
        Normalized DomiRank centrality of the nodes in the graph G. 
    '''
    
    N = G.number_of_nodes()

    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        G = nx.to_scipy_sparse_array(G).astype(float) #convert to scipy sparse if it is a graph 
    else:
        G = G.copy()
        
    gamma = np.zeros(N, dtype = float)

    # if sigma is not provided, we set it to the upper limit of the convergence interval
    if sigma == -1: 
        eigenvalues, _ = eigsh(G)
        min = np.min(eigenvalues)
        del eigenvalues, _
        if min == 0:
            print("Warning in domirank: The maximum eigenvalue of the graph is zero. Returning zero vector.")
            return gamma
        sigma = -1/min

    # the convergence condition is ||gamma - gamma_prev|| < epsilon*N*dt
    convergence = epsilon*N*dt    

    for i in range(maxsteps):
        aux = (sigma*G @ (1-gamma) - gamma)*dt #f(t, gamma(t))*dt
        
        if method == "heun": # if we want more accuracy...
            gamma2 = gamma + aux # gamma*(t+dt) = gamma(t) + dt*f(t, gamma(t))
            gamma +=  0.5*(aux + dt*(sigma*(G @ gamma2) - gamma2)) # gamma(t+dt) = gamma(t) + dt/2*(f(t, gamma(t)) + f(t+dt, gamma*))
        
        else: #euler
            gamma += aux #gamma(t+dt) = gamma(t) + dt*f(t, gamma(t))

        if np.linalg.norm(aux) < convergence and i > 0:
            print(f'Convergence reached after {i} steps.')
            break

        
    return gamma/np.max(gamma) #normalize to [0,1]

def domirank_paper(G, sigma = -1, dt = 0.1, epsilon = 1e-6, maxIter = 100000, checkStep = 10):
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
    pGAdj = sigma*G.astype(np.float32)
    Psi = np.zeros(pGAdj.shape[0]).astype(np.float32)
    maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float32)
    dt = np.float32(dt)
    j = 0
    boundary = epsilon*pGAdj.shape[0]*dt
    for i in range(maxIter):
        tempVal = ((pGAdj @ (1-Psi)) - Psi)*dt 
        Psi += tempVal.real
        if i% checkStep == 0:
            if np.abs(tempVal).sum() < boundary:
                break
            maxVals[j] = tempVal.max()
            if i == 0:
                initialChange = maxVals[j]
            if j > 0:
                if maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
                    return False, Psi
            j+=1

    return True, Psi/np.max(Psi) #normalize to [0,1]

G = nx.grid_2d_graph(7, 7, periodic=False) #create a grid graph
G2 = nx.to_scipy_sparse_array(G).astype(float)
eigenvalues, _ = eigsh(G2)
min = np.min(eigenvalues)
del eigenvalues, _

for sigma in [ 0.01, 0.75, 0.95, 0.99]:
    b, gamma = domirank_paper(G, sigma=-sigma/min)
    for node in G.nodes():
        G.nodes[node]['domirank'] = gamma[list(G.nodes()).index(node)]
    pos = nx.spring_layout(G, seed=200)
    nx.draw(G, pos, cmap=plt.get_cmap('cividis'), node_color=gamma, font_color='white')
    plt.savefig(f"grid_sigma_{sigma}_paper.png")
    plt.clf()