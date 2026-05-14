import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import functions as f

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

N = 100
G = nx.erdos_renyi_graph(N, 0.1)
steps = 200
lams = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
dt = 0.01
theta = initialize_random(N)


for lam in lams:
    theta = initialize_random(N)
    omegas = initialize_random(N)
    r = evolve_kuramoto(theta, G, lam, omegas, dt, steps)
    plt.plot(r, label = f"λ={lam}")
    print(f"Final order parameter for λ={lam}: {r[-1]}")
plt.legend()
plt.xlabel("Time steps")
plt.ylabel("Order parameter r")
plt.show()