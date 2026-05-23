import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import functions as f
from scipy.sparse.linalg import eigsh


def order_parameter(theta):
    '''
    Calculate the order parameter r of the Kuramoto model given the theta vector
    '''
    return np.abs(np.mean(np.exp(1j * theta)))

def thetaDot(theta, rows, cols, w, omega):
    diff = theta[cols] - theta[rows]
    interaction = w * np.sin(diff)

    dtheta = np.bincount(rows, weights=interaction, minlength=len(theta))

    return omega + dtheta

def rk4(theta, rows, cols, w, omega, dt):
    '''
    Runge-Kutta 4th order method for numerical integration of the Kuramoto model
    '''
    k1 = thetaDot(theta, rows, cols, w, omega)*dt
    k2 = thetaDot(theta + 0.5*k1, rows, cols, w, omega)*dt
    k3 = thetaDot(theta + 0.5*k2, rows, cols, w, omega)*dt
    k4 = thetaDot(theta + k3, rows, cols, w, omega)*dt
    
    return theta + (1/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def evolve_kuramoto(theta, omega, dt, rows, cols, w , steps = 1000, epsilon = 1e-4, sampling = 0, retrace=False, break_on_convergence=True):
    '''
    Evolve the Kuramoto model for a given number of steps
    if retrace is true, it will retrace the steps backwards
    '''

    if sampling == 0: # sample every 1% of the steps by default
        sampling = int(steps/100)
        if sampling == 0: # if the number of steps is too small, we sample every step
            sampling = 1
    
    active = w != 0
    active_rows = rows[active]
    active_cols = cols[active]
    active_w = w[active]        
    r = []

    window = int(steps/10) # we check the convergence of r in a window of 10% of the steps
    conv_iter = steps
    for _ in range(steps):
        theta = rk4(theta, active_rows, active_cols, active_w, omega, dt)
        if _ % sampling == 0:
            r.append(order_parameter(theta))
            if len(r) > window and np.std(r[-window:]) < epsilon and conv_iter == steps: # if r converges, we stop the evolution
                conv_iter = _
                if break_on_convergence:
                    break
        
    if retrace:
        for _ in range(steps):
            theta = rk4(theta, active_rows, active_cols, active_w, omega, -dt)
            if _ % sampling == 0:
                r.append(order_parameter(theta))
                if len(r) > window and np.std(r[-window:]) < epsilon and conv_iter == steps: # if r converges, we stop the evolution
                    conv_iter = _
                    if break_on_convergence:
                        break
    return r, theta, conv_iter

def initialize_random(N, limits=(0, 2*np.pi)):
    '''
    Returns a dict where keys are node ids and values are random angles between 0 and 2*pi, for a graph with N nodes
    '''
    return np.random.uniform(limits[0], limits[1], N)


def change_edge_weight(w, rows, cols, node_list, weight):
    '''
    Change the weight of an edge in a graph G sparray, given the edge and the new weight
    '''
    node_list = np.atleast_1d(node_list)
    mask = np.isin(rows, node_list) | np.isin(cols, node_list)
    w[mask] = weight
    return w

def kuramoto_recovery_step(to_recover, rows, cols, w, lam, p, sampling =0, method="random"):
    '''
    given the lists of G links (in rows and cols), their weights w and the list of nodes to recover (w=0)
    we recover one node with probability p according to method by assigning its links a weight of lam, 
    and return the new weights w and a check variable that is 1 if we have recovered a node and 0 otherwise
    '''

    global_check = 0
    node_list = []
    for s in range(sampling):
        if len(to_recover) == 0: # if there are no nodes to recover, we skip the recovery process
            break
        
        chosen, check = f.choose_recovered_node(to_recover, method, p)
        
        if check == 1: # if we have chosen a node to recover, we recover it in the graph
            global_check += 1
            node_list.append(chosen)
            to_recover.remove(chosen) # we remove the node from the list of nodes to recover, as we have recovered it in the graph
            
    w = change_edge_weight(w, rows, cols, node_list, lam) 

    return w, global_check

def kuramoto_recovery(to_recover, theta, rows, cols, w, lam, omega, dt, steps, p, epsilon = 1e-4, sampling =0, sampling_kura=0, method="random", centrality_func=None):
    '''
    given the list of nodes from G_original to recover, the current graph GAdj,
    we recover the nodes with probability p according to method, 
    and save the order parameter, links and component size after every recovery step, 
    until we have recovered all the nodes in to_recover.
    '''

    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(len(to_recover)/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1
    if sampling_kura == 0:
        sampling_kura = int(len(to_recover)/100)
        if sampling_kura == 0: 
            sampling_kura = 1
    r = []
    r_stable = []
    while len(to_recover) > 0: # we want to recover all the nodes
        w, check = kuramoto_recovery_step(to_recover, rows, cols, w, lam, p, sampling=sampling, method=method)
        if check > 0: # a node has been recovered (sampling is included in kuramoto_recovery_step)
            # as the net has changed, we evolve the dynamics
            r_aux, theta, _ = evolve_kuramoto(theta, omega, dt, rows, cols, w, steps=steps, epsilon=epsilon, sampling=sampling_kura)
            r.extend(r_aux) # and we save the kuramoto trajectory
        if len(r) > 1: # the order parameter after each recovery step is the last one of r_aux, as we evolve the dynamics after each recovery step
            # if the net hasnt changed, has stayed the same as we dont enter kuramoto evolution, but if it has, r has changed accordinlgy
            r_stable.append(r[-1])
        # print(f'{len(to_recover)-1} left to recover')
    return r, r_stable

def kuramoto_attack(theta, omega, G, dt, steps, attackStrategy = [], sampling =0, sampling_kura = 0, epsilon = 1e-4, centrality_func=None):
    '''
    attack the network and compute the order parameter, links and component size after each attack step, 
    until we have attacked all the nodes in attackStrategy.
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph
    else:
        GAdj = G.copy()
    N = len(theta)
    rows, cols = GAdj.nonzero()
    w = GAdj.data

    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(N/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1

    if sampling_kura == 0:
        sampling_kura = int(N/100)
        if sampling_kura == 0: 
            sampling_kura = 1


    r = []
    r_before_attack = []
    node_map = {i: i for i in range(N)} # we need to keep track of the node labels as they change after each attack, as the attack strategy is based on the original labels
    r_temporal, theta, _ = evolve_kuramoto(theta, omega, dt, rows, cols, w, steps=steps, epsilon=epsilon, sampling=sampling_kura)
    r_before_attack.append(r_temporal[-1]) # we save the order parameter before the attack to see the effect of the attack on the order parameter
    
    for i in range(N-1):
        # we evolve the Kuramoto model after each attack to see the effect on the order parameter and sinchronization
        if i%sampling == 0: # we also want to save the original state:
            if i != 0:
                attacked, node_map = f.changing_attack(GAdj, attackStrategy=attackStrategy, centrality_func=centrality_func, node_map=node_map, sampling=sampling, i=i) 
                w = change_edge_weight(w, rows, cols, attacked, 0)
                r_more, theta, _ = evolve_kuramoto(theta, omega, dt, rows = rows, cols = cols, w = w, steps = steps, epsilon = epsilon, sampling = sampling_kura)
                r.extend(r_more) 
                r_before_attack.append(r_more[-1]) # we save the order parameter before the attack to see the effect of the attack on the order parameter      
    
    return r, r_before_attack
    
def kuramoto_attack_recovery(theta, lam, omega, G, dt, steps, p,attackStrategy = [],  sampling =0, centrality_func=None, method="sequential", epsilon = 1e-4, sampling_kura = 0):
    '''
    attack the network and recover it at the same time, computing the order parameter, links and component size after each attack and recovery step,
    until we have attacked all the nodes in attackStrategy and recovered all the nodes in to_recover.
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph
    else:
        GAdj = G.copy()

    rows, cols = GAdj.nonzero()
    w = GAdj.data

    N = len(theta)

    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(N/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1

    if sampling_kura == 0:
        sampling_kura = int(N/100)
        if sampling_kura == 0: 
            sampling_kura = 1

    r = []
    to_recover = []
    r_before_attack = []
    node_map = {i: i for i in range(N)} # we need to keep track of the node labels as they change after each attack, as the attack strategy is based on the original labels
    # first evolution of the model, we need to enter the attack loop in the steady state
    r_temporal, theta, _ = evolve_kuramoto(theta, omega, dt, rows, cols, w, steps=steps, epsilon=epsilon, sampling=sampling_kura)
    r_before_attack.append(r_temporal[-1]) # we save the order parameter before any attacks
    
    for i in range(N-1):
        # we evolve the Kuramoto model after each attack to see the effect on the order parameter and sinchronization
        if i%sampling == 0: # we also want to save the original state
            if i != 0:
                # in the same time step we have to attack and recover, so evolve_kuramoto is last
                # checking the attack method
                attacked, node_map = f.changing_attack(GAdj, attackStrategy=attackStrategy, centrality_func=centrality_func, node_map=node_map, sampling=sampling, i=i)
                
                # the attack itself
                w = change_edge_weight(w, rows, cols, attacked, 0) # we remove the edges to the attacked node, set the weight to 0
                to_recover.extend(attacked)

                # now we recover some nodes
                w, check = kuramoto_recovery_step(to_recover, rows, cols, w, lam, p, sampling=sampling, method=method)

                # finally we evolve the dynamics to see the effect of the attack and recovery on the order parameter
                # we dont need the recovery check, as a change is made either way bc of the attack
                r_aux, theta, _ = evolve_kuramoto(theta, omega, dt, rows = rows, cols = cols, w = w, steps = steps, epsilon = epsilon, sampling=sampling_kura)
                r.extend(r_aux)
                r_before_attack.append(r[-1]) # save the order parameter after the attack == before the next attack step...
                  
    # now we just recover        
    r_recovered, r_stable= kuramoto_recovery(to_recover, theta, rows, cols, w, lam, omega, dt, steps, p, epsilon=epsilon, sampling_kura=sampling_kura, method=method)
    r.extend(r_recovered)
    r_before_attack.extend(r_stable) # we save the order parameter before the attack to see the effect of the attack on the order parameter
    

    return r, r_before_attack

def calculate_area(r, dt):
    '''
    Calculate the area under the curve of r as a function of time, given the time step dt
    '''
    area = 0
    for i in range(1, len(r)):
        area += (r[i]+r[i-1])/2*dt
    return area



