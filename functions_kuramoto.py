import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import functions as f
from scipy.sparse.linalg import eigsh


def order_parameter(theta):
    '''
    Calculate the order parameter r of the Kuramoto model given the theta vector
    '''
    theta_values = list(theta.values())
    return np.sqrt((np.sum(np.cos(theta_values))**2 + np.sum(np.sin(theta_values))**2))/len(theta_values)

def thetaDot(theta, G, omega):
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
        new_theta[i] = omega[i] + G.edge[i]['weight']*np.sum([np.sin(theta[j]-theta[i]) for j in GAdj.neighbors(i)])
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
    theta_values = list(theta.values())
    k1 = thetaDot(theta_values, G, lam, omega)*dt
    k2 = thetaDot(theta_values + 0.5*k1, G, lam, omega)*dt
    k3 = thetaDot(theta_values + 0.5*k2, G, lam, omega)*dt
    k4 = thetaDot(theta_values + k3, G, lam, omega)*dt

    theta_values += (1/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    #theta = check_angles(theta)
    
    return {i: theta_values[i] for i in range(len(theta_values))}

def evolve_kuramoto(theta, G, lam, omega, dt, steps, epsilon = 1e-4, retrace=False):
    '''
    Evolve the Kuramoto model for a given number of steps
    if retrace is true, it will retrace the steps backwards
    '''
    GAdj = G.copy()

    r = []
    for _ in range(steps):
        theta = rk4(theta, GAdj, lam, omega, dt)
        r.append(order_parameter(theta))
        if len(r) > 1 and np.linalg.norm(r[-1]-r[-2]) < epsilon: # if r converges, we stop the evolution
            conv_iter = _
        
    if retrace:
        for _ in range(steps):
            theta = rk4(theta, GAdj, lam, omega, -dt)
            r.append(order_parameter(theta))
            if len(r) > 1 and np.linalg.norm(r[-1]-r[-2]) < epsilon: # if r converges, we stop the evolution
                conv_iter = _
    return r, theta, conv_iter

def initialize_random(N):
    '''
    Returns a dict where keys are node ids and values are random angles between 0 and 2*pi, for a graph with N nodes
    '''
    dictionary = {i: np.random.uniform(0, 2*np.pi) for i in range(N)}
    return dictionary


def change_edge_weight(G, node_list, weight):
    '''
    Change the weight of an edge in a graph G, given the edge and the new weight
    '''
    if type(node_list) != list:
        node_list = [node_list]
    for node in node_list:
        for neighbor in G.neighbors(node):
            G[node][neighbor]['weight'] = weight

    return G

def kuramoto_recovery_step(to_recover, theta, GAdj, lam, omega, dt, steps, p, epsilon = 1e-4,sampling =0, method="random"):
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
        
        chosen, check = f.choose_recovered_node(to_recover, method, p)
        if check == 1: # if we have chosen a node to recover, we recover it in the graph
            GAdj = change_edge_weight(GAdj, chosen, lam) 
            
            for j in range(steps):
                # as the network has changed, we need to evolve the dynamics
                theta = rk4(theta, GAdj, lam, omega, dt)
                if j % sampling == 0: # save data every sampling steps
                    r.append(order_parameter(theta))
                    components.append(f.get_component_size(GAdj))
                    links.append(f.get_link_size(GAdj))
                    if len(r)>1 and np.linalg.norm(r[-1]-r[-2]) < epsilon: # if r converges, we move on
                        break
        elif len(r) > 0: # if we haven't chosen a node to recover, we keep the same graph and evolve the dynamics to see if it converges, but we don't save the data as we haven't recovered any node
            r.append(r[-1]) # if no nodes have been recovered, we keep the same order parameter 
            links.append(links[-1]) # we also keep the same number of links and component size
            components.append(components[-1])
        else:
            r.append(order_parameter(theta)) # if we haven't recovered any node yet, we save the initial state
            links.append(f.get_link_size(GAdj))
            components.append(f.get_component_size(GAdj))

    return r, links, components

def kuramoto_recovery(to_recover, G_original, theta, GAdj, lam, omega, dt, steps, p, epsilon = 1e-4, sampling =0, sampling_kura=0, method="random"):
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
    if sampling_kura == 0:
        sampling_kura = int(len(to_recover)/100)
        if sampling_kura == 0: 
            sampling_kura = 1

    while len(to_recover) > 0: # we want to recover all the nodes
        r, links, components = kuramoto_recovery_step(to_recover, theta, GAdj, lam, omega, dt, steps, p, epsilon, sampling, method)
    return r, links/initialLinks, components/initialComponent

def kuramoto_attack(theta, G, lam, omega, dt, steps, attackStrategy = [], sampling =0, sampling_kura = 0, epsilon = 1e-5, centrality_func=None):
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
    node_map = dict(zip(range(N), GAdj.nodes())) # we need to keep track of the node labels as they change after each attack, as the attack strategy is based on the original labels

    for i in range(N-1):
        # we evolve the Kuramoto model after each attack to see the effect on the order parameter and sinchronization
        if i%sampling == 0: # we also want to save the original state
            for j in range(steps): # kurmaoto dynamics evolution
                theta = rk4(theta, GAdj, lam, omega, dt)
                if j % sampling_kura == 0: # save data every sampling steps
                    r.append(order_parameter(theta))
                    links.append(f.get_link_size(GAdj)/initialLinks)
                    components.append(f.get_component_size(GAdj)/initialComponent)
                    runtime += 1
                if len(r) > 1 and np.linalg.norm(r[-1]-r[-2]) < epsilon: # if r converges, we move on
                    break
            # once the system has evolved, we attack the network by removing the coupling strength
            if i != 0:
                attacked, node_map = f.changing_attack(GAdj, attackStrategy=attackStrategy, centrality_func=centrality_func, node_map=node_map, sampling=sampling, i=i) 
                GAdj = change_edge_weight(GAdj, attacked, 0)
                  
    
    return r, links, components
    
def kuramoto_attack_recovery(theta, G, lam, omega, dt, steps, p,attackStrategy = [],  sampling =0, centrality_func=None, method="random", epsilon = 1e-4, sampling_kura = 0):
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
    node_map = dict(zip(range(N), GAdj.nodes())) # we need to keep track of the node labels as they change after each attack, as the attack strategy is based on the original labels

    for i in range(N-1):
        # we evolve the Kuramoto model after each attack to see the effect on the order parameter and sinchronization
        if i%sampling == 0: # we also want to save the original state
            r_aux, links_aux, components_aux = kuramoto_recovery_step(to_recover, theta, GAdj, lam, omega, dt, steps, p, epsilon, sampling, method)
            r.extend(r_aux)
            links_aux = [links_aux[k]/initialLinks for k in range(len(links_aux))]
            components_aux = [components_aux[k]/initialComponent for k in range(len(components_aux))]
            links.extend(links_aux)
            components.extend(components_aux)
            if i != 0: # attack the network every sampling steps, but not at the beginning, as we want to save the original state
                attacked, node_map = f.changing_attack(GAdj, attackStrategy=attackStrategy, centrality_func=centrality_func, node_map=node_map, sampling=sampling, i=i)
                GAdj = change_edge_weight(GAdj, attacked, 0) # we remove the edges to the attacked node,
                to_recover.extend(attacked)
                 
    r_recovered, links_recovered, components_recovered = kuramoto_recovery(to_recover, G, theta, GAdj, lam, omega, dt, steps, p, epsilon, sampling_kura, method)
    r.extend(r_recovered)
    links.extend(links_recovered)
    components.extend(components_recovered)

    return r, links, components




