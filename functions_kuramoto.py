import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import functions as f
from scipy.sparse.linalg import eigsh

def order_parameter(theta):
    '''
    Calculate the order parameter r of the Kuramoto model given the theta vector
    '''
    return np.sqrt((np.sum(np.cos(theta))**2 + np.sum(np.sin(theta))**2))/len(theta)

def thetaDot(theta, G, lam, omega):
    '''
    time derivative of the theta vector in the Kuramoto model, 
    given a graph G, coupling strength lam and natural frequencies omega
    '''
    
    if type(G) != nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
    else:
        GAdj = G.copy()
    new_theta = np.zeros(len(theta))
    for i in range(len(theta)):
        new_theta[i] = omega[i] + lam*np.sum([np.sin(theta[j]-theta[i]) for j in GAdj.neighbors(i)])
    return new_theta

def check_angles(theta):
    '''
    Check if the angles are between 0 and 2*pi, if not, bring them back to that range
    '''
    for i in range(len(theta)):
        resto = theta[i] - int(theta[i]/(2*np.pi))*(2*np.pi)
        if resto < 0:
            resto += 2*np.pi
        theta[i] = resto
    return theta

def rk4(theta, G, lam, omega, dt):
    '''
    Runge-Kutta 4th order method for numerical integration of the Kuramoto model
    '''
    k1 = thetaDot(theta, G, lam, omega)*dt
    k2 = thetaDot(theta + 0.5*k1, G, lam, omega)*dt
    k3 = thetaDot(theta + 0.5*k2, G, lam, omega)*dt
    k4 = thetaDot(theta + k3, G, lam, omega)*dt

    theta += (1/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    #theta = check_angles(theta)
    
    return theta + (1/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def evolve_kuramoto(theta, G, lam, omega, dt, steps, retrace=False):
    '''
    Evolve the Kuramoto model for a given number of steps
    if retrace is true, it will retrace the steps backwards
    '''
    GAdj = G.copy()

    r = []
    for _ in range(steps):
        theta = rk4(theta, GAdj, lam, omega, dt)
        r.append(order_parameter(theta))
    if retrace:
        for _ in range(steps):
            theta = rk4(theta, GAdj, lam, omega, -dt)
            r.append(order_parameter(theta))
    return r

def initialize_random(N):
    '''
    Returns a vector initialized with random values between 0 and 2*pi
    '''
    return np.random.uniform(0, 1, N)*2*np.pi


def kuramoto_recovery_step(to_recover, G_original, theta, GAdj, lam, omega, dt, steps, p, epsilon = 1e-4,sampling =0, centrality_func=None, method="random"):
    '''
    given the list of nodes from G_original to recover, the current graph GAdj,
    we recover the nodes with probability p according to method,
    and save the order parameter,links and component size (not normalized) after every recovery step, 
    until we have run sampling steps.
    '''

    r = []
    components = []
    links = []  
    for s in range(sampling):
        if len(to_recover) == 0: # if there are no nodes to recover, we skip the recovery process
            break
        check = f.recovery_step(to_recover, GAdj, method, p, G_original) 
        
        if check == 1: # nodes have been recovered
            for j in range(steps):
                # as the network has changed, we need to evolve the dynamics
                theta = rk4(theta, GAdj, lam, omega, dt)
                if j % sampling == 0: # save data every sampling steps
                    r.append(order_parameter(theta))
                    components.append(f.get_component_size(GAdj))
                    links.append(f.get_link_size(GAdj))
                    if len(r)>1 and np.linalg.norm(r[-1]-r[-2]) < epsilon: # if r converges, we move on
                        break
        else:
            r.append(r[-1]) # if no nodes have been recovered, we keep the same order parameter 
            links.append(links[-1]) # we also keep the same number of links and component size
            components.append(components[-1])

    return r, links, components

def kuramoto_recovery(to_recover, G_original, theta, GAdj, lam, omega, dt, steps, attackStrategy, p, r = [], links = [], components = [], epsilon = 1e-4,sampling =0, centrality_func=None, method="random"):
    '''
    given the list of nodes from G_original to recover, the current graph GAdj,
    we recover the nodes with probability p according to method, 
    and save the order parameter, links and component size after every recovery step, 
    until we have recovered all the nodes in to_recover.
    '''
    initialLinks = float(f.get_link_size(G_original))   
    initialComponent = float(f.get_component_size(G_original))
    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(len(to_recover)/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1

    while len(to_recover) > 0: # we want to recover all the nodes
        r, links, components = kuramoto_recovery_step(r, links, components, to_recover, G_original, theta, GAdj, lam, omega, dt, steps, p, epsilon, sampling, centrality_func, method)
    return r, links/initialLinks, components/initialComponent

def kuramoto_attack(theta, G, lam, omega, dt, steps, attackStrategy, sampling =0, sampling_kura = 0, epsilon = 1e-4):
    '''
    attack the network and compute the order parameter, links and component size after each attack step, 
    until we have attacked all the nodes in attackStrategy.
    '''
    GAdj = G.copy()
    N = nx.number_of_nodes(GAdj)
    initialComponent = float(f.get_component_size(GAdj)) # for normalization to lcc(0) = 1
    initialLinks = float(f.get_link_size(GAdj))

    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(N/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1

    if sampling_kura == 0:
        sampling_kura = int(N/100)
        if sampling_kura == 0: 
            sampling_kura = 1

    components = []
    links = []
    r = []

    for i in range(N-1):
        # we evolve the Kuramoto model after each attack to see the effect on the order parameter and sinchronization
        if i%sampling == 0: # we also want to save the original state
            for j in range(steps):
                theta = rk4(theta, GAdj, lam, omega, dt)
                if j % sampling_kura == 0: # save data every sampling steps
                    r.append(order_parameter(theta))
                    links.append(f.get_link_size(GAdj)/initialLinks)
                    components.append(f.get_component_size(GAdj)/initialComponent)
                if len(r) > 1 and np.linalg.norm(r[-1]-r[-2]) < epsilon: # if r converges, we move on
                    break
            if i != 0:
                GAdj = f.remove_node(GAdj, attackStrategy[i-sampling:i])
                del theta[attackStrategy[i-sampling:i]] 
                del omega[attackStrategy[i-sampling:i]]

    
    return r, links, components
    
def kuramoto_attack_recovery(theta, G, lam, omega, dt, steps, attackStrategy, p, sampling =0, centrality_func=None, method="random", epsilon = 1e-4):
    '''
    attack the network and recover it at the same time, computing the order parameter, links and component size after each attack and recovery step,
    until we have attacked all the nodes in attackStrategy and recovered all the nodes in to_recover.
    '''
    if type(G) != nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph
    else:
        GAdj = G.copy()

    N = nx.number_of_nodes(GAdj)
    initialComponent = float(f.get_component_size(GAdj)) # for normalization to lcc(0) = 1
    initialLinks = float(f.get_link_size(GAdj))

    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(N/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1

    if sampling_kura == 0:
        sampling_kura = int(N/100)
        if sampling_kura == 0: 
            sampling_kura = 1

    components = []
    links = []
    r = []
    to_recover = []

    for i in range(N-1):
        # we evolve the Kuramoto model after each attack to see the effect on the order parameter and sinchronization
        if i%sampling == 0: # we also want to save the original state
            r_aux, links_aux, components_aux = kuramoto_recovery_step(r, links, components, to_recover, G, theta, GAdj, lam, omega, dt, steps, p, epsilon, sampling_kura, centrality_func, method)
            r.extend(r_aux)
            links.extend(links_aux/initialLinks)
            components.extend(components_aux/initialComponent)
            if i != 0: # attack the network every sampling steps, but not at the beginning, as we want to save the original state
                GAdj = f.remove_node(GAdj, attackStrategy[i-sampling:i])
                to_recover.extend(attackStrategy[i-sampling:i])
    
    r_recovered, links_recovered, components_recovered = kuramoto_recovery(r, links, components, to_recover, G, theta, GAdj, lam, omega, dt, steps, p, epsilon, sampling_kura, centrality_func, method)
    r.extend(r_recovered)
    links.extend(links_recovered)
    components.extend(components_recovered)

    return r, links, components





N = 100
G = nx.erdos_renyi_graph(N, 0.1)
steps = 200
lams = [1.0]
dt = 0.01


for lam in lams:
    theta = initialize_random(N)
    omegas = initialize_random(N)
    eigvals, _ = eigsh(nx.to_scipy_sparse_array(G).astype(float))
    eig = np.min(eigvals)
    psi = f.domirank(G, sigma=-0.5/eig)
    attack = f.generate_attack(psi)
    r, links, components = kuramoto_attack(theta, G, lam, omegas, dt, steps, attackStrategy = attack)
    plt.plot(r, label = f"λ={lam}")
    print(f"Final order parameter for λ={lam}: {r[-1]}")
plt.legend()
plt.xlabel("Time steps")
plt.ylabel("Order parameter r")
plt.show()