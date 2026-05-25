import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.sparse.linalg import eigsh
import functions_kuramoto as fk
import functions as f
from joblib import Parallel, delayed
from multiprocessing import freeze_support



def make_graph(name, N, k):
    if name == 'BA':  
        G = nx.barabasi_albert_graph(N, int(k/2), seed=42)
    if name == 'ER':
        G = nx.erdos_renyi_graph(N, p=k/(N-1), seed=42)

    eig = eigsh(nx.to_scipy_sparse_array(G).astype(float), return_eigenvectors=False)
    return G, np.min(eig)

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def evolve(dt, k_list, avg_, net_name, list_lambdas=np.flip(np.arange(0.1, 1.6, 0.1))):
    steps = 100
    N=100
    print(f'dt={dt} starting')
    for k in k_list:
        G = nx.barabasi_albert_graph(N, int(k/2), seed=42)
        G = f.relabel_nodes(G)
        
        print(f'k={k} starting')
    
        for lam in list_lambdas:
            nx.set_edge_attributes(G, lam, 'weight')
            conv_iter = 0
            t_init = time.time()
            if k<=6:
                avg = avg_[1]
            else:
                avg = avg_[0]
            r_total = 0
            base_out = f'results_kuramoto/{net_name}_{N}/k{k}_dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/attack/'
            if not os.path.exists(base_out):
                os.makedirs(base_out)
            out = base_out + f'{lam:.1f}'
            if os.path.isfile(out + '_results.txt'):
                print(f'lambda={lam:.1f} already done, skipping')
                continue
            # average over different initial conditions to remove fluctuations
            for _ in range(avg):
                theta = fk.initialize_random(N)
                omegas = fk.initialize_random(N)
                r, _, conv_step = fk.evolve_kuramoto(theta, G, omegas, dt, steps, break_on_convergence=False)
                print(conv_step)
                if _ == 0:
                    r_total = np.array(r)
                    
                r_total += np.array(r)
                if np.any(np.asarray(r) > 1.):
                    print(f'r integration not correct for lambda={lam:.1f}, dt={dt}, k={k}!')
                conv_iter += conv_step
            r_total /= avg
            conv_step = conv_iter/avg
            
            

            # plotting the evolution of r
            
            done = time.time() - t_init
            
            np.savetxt(out + '_results.txt', np.array(r_total), fmt='%.4f')
            file = open(out + '_convergence.txt', 'w')
            file.write(f'runtime (s)\t\t{done:.0f}\nfinal r_err\t\t{np.linalg.norm(r_total[-1]-r_total[-2]):.5f}\nConvergence time\t\t{conv_step*dt:.2f}\n')
            file.close()

            print(f'lambda={lam} done in {done:.2f} seconds, convergence time: {conv_step*dt:.2f} time units, final r={r_total[-1]:.4f}')
    return

def one_simulation(G, N, lam, dt, steps, attackStrategy, do, p, sampling, samplig_kura, centrality_func, recovery_method):
    nx.set_edge_attributes(G, lam, 'weight')
    theta = fk.initialize_random(N)
    omega = fk.initialize_random(N, (-0.5, 0.5))

    if do == 'attack':
        r, r_stable_aux = fk.kuramoto_attack(theta, omega, G, dt, steps, attackStrategy, sampling = sampling, sampling_kura = samplig_kura, centrality_func=centrality_func)
    elif do == 'recover':
        r, r_stable_aux = fk.kuramoto_attack_recovery(theta, lam, omega, G, dt, steps, p=p, attackStrategy = attackStrategy, sampling = sampling, sampling_kura = samplig_kura, centrality_func=centrality_func, method=recovery_method)
    else: 
        raise ValueError('do must be either attack or recover')
    return r, r_stable_aux, np.trapz(r), np.trapz(r_stable_aux)

def run_lambda(G, N, lam, dt, steps, avg, attackStrategy, do, p, sampling, sampling_kura, centrality_func, recovery_method):
    r_all = []
    r_stable_all = []
    areas_all = []
    areas_stable = []

    for _ in range(avg):
        r, r_stable, area_r, area_stable = one_simulation(G, N, lam, dt, steps, attackStrategy, do, p, sampling, sampling_kura, centrality_func, recovery_method)
        r_all.append(r)
        r_stable_all.append(r_stable)
        areas_all.append(area_r)
        areas_stable.append(area_stable)

    return r_all, r_stable_all, areas_all, areas_stable

def attack_or_recover(task_id, name, N, k, sigma, eig, strategy_name, do = 'attack', p =0.25, dt = 0.1, steps = 100, avg = 10, list_lambdas=np.arange(0.05, 1.1, 0.05), sampling = 0, sampling_kura = 0, centrality_func = None, recovery_method = 'sequential'):
    print("ENTER TASK", task_id, os.getpid())

    G, eig = make_graph(name, N, k)

    base_dir = f'results_kuramoto/_{name}_{N}_k{k}/'
    if do == 'attack':
        base_out = base_dir + f'dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}_{do}/{strategy_name}_s{-sigma*eig:.2f}/'
    elif do == 'recover':
        base_out = base_dir + f'dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}_{do}_{recovery_method}_p{p}/{strategy_name}_s{-sigma*eig:.2f}/'
    else:
        raise ValueError('do must be either attack or recover')
    
    if not os.path.exists(base_out):
        os.makedirs(base_out)
    if strategy_name == 'domirank':
        psi = f.domirank(G, sigma)
        attackStrategy = f.generate_attack(psi)
    
    areas_mean, areas_std = [], []
    areas_stable_mean, areas_stable_std = [], []

    for lam in list_lambdas:
        print(f'{task_id}, {os.getpid()} running with lambda={lam:.3f}, sigma={-sigma*eig:.2f}')
        r_evol, r_stable_all, areas_all, areas_stable = run_lambda(G, N, lam, dt, steps, avg, attackStrategy, do, p, sampling, sampling_kura, centrality_func, recovery_method)
        r_stable_mean, r_stable_std = tolerant_mean(r_stable_all)
        areas_mean.append(np.mean(areas_all))
        areas_std.append(np.std(areas_all))

        areas_stable_mean.append(np.mean(areas_stable))
        areas_stable_std.append(np.std(areas_stable))

        r_evol_mean, r_evol_std = tolerant_mean(r_evol)

        np.savetxt(base_out + f'{lam:.3f}_r_evolution.txt', np.column_stack((r_evol_mean, r_evol_std)), fmt=['%.5f', '%.5e'], delimiter = '\t')
        np.savetxt(base_out + f'{lam:.3f}_r_stable.txt', np.column_stack((r_stable_mean, r_stable_std)), fmt=['%.5f', '%.5e'], delimiter = '\t')
    
    np.savetxt(base_out + 'areas.txt', np.column_stack((list_lambdas, areas_mean, areas_std)), fmt=['%.3f','%.5f', '%.5e'], delimiter = '\t')
    np.savetxt(base_out + 'areas_stable.txt', np.column_stack((list_lambdas, areas_stable_mean, areas_stable_std)), fmt=['%.3f','%.5f', '%.5e'], delimiter = '\t')

    return



def make_task(name, mode, N, k, sigma, eig, p, avg):
    return delayed(attack_or_recover)(
        task_id = f"{name}_{mode}_{N}_{k}_{-sigma*eig:.2f}_{avg}_{0.1}",
        name=name,
        N=N,
        k=k,
        sigma=sigma,
        eig=eig,
        strategy_name="domirank",
        do=mode,
        p=p,
        steps=N,
        avg=avg,
        sampling=1,
        sampling_kura=1,
    )


if __name__ == "__main__":
    freeze_support()

    N = 100
    p = 0.25
    k = 6
    avg = 10

    G_ba, eig_ba = make_graph('BA', N, k)
    G_er, eig_er = make_graph('ER', N, k)
    graphs = {
        "ER": (G_er, eig_er),
    "BA": (G_ba, eig_ba)
    }
    
    sigma_list = np.arange(0.03, 1.02, 0.03)

    tasks = []
    
    for name in ["ER", "BA"]:
        optimal, _ =f.old_optimal_sigma(nx.to_scipy_sparse_array(graphs[name][0]), graphs[name][1])
        #sigma_list = sigma_list_og + [-optimal*graphs[name][1]]
        for sigma in sigma_list:
                eig = graphs[name][1]
                tasks.append(make_task(name, "attack", N, k, -sigma/eig, eig, p, avg=avg))
                tasks.append(make_task(name, "recover", N, k, -sigma/eig, eig, p, avg))
    
    Parallel(n_jobs=6, backend="loky")(tasks)
