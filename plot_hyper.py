import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import functions_hyper as fh
import hypernetx as hnx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx

def plotting_psi(H, psi, node_names, ax, layout='spring', pos=None):

    ax.clear()

    psi = psi / np.max(psi)

    psi_dict = dict(zip(node_names, psi))
    psi_ordered = [psi_dict[n] for n in H.nodes]

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.cividis

    node_colors = [cmap(norm(v)) for v in psi_ordered]

    G = H.bipartite()

    if pos is None:
        if layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

    hnx.draw(
        H,
        ax=ax,
        pos=pos,
        nodes_kwargs={"facecolors": node_colors}
    )

    ax.set_title("Hypergraph evolution")

    return pos


def plot_animation(H, node_names, file_root, eta, scale=1):

    try:

        data = np.loadtxt(f'{file_root}_evolution.txt', dtype=float)
        conv = np.loadtxt(f'{file_root}_convergence.txt', dtype=float)
        thetas = conv[:, 0]
        steps = conv[:, 1].astype(int)
        total_steps = [i for i in range(len(data))]

        idx = np.where(thetas == eta)[0][0]

        frames = total_steps[steps[idx-1]:steps[idx-1]+steps[idx]]

        print(f"Creating animation for eta={eta} with {frames} frames.")

        fig, ax = plt.subplots(figsize=(8, 6))

        # posición fija para evitar que el layout cambie
        G = H.bipartite()
        pos = nx.spring_layout(G, seed=42)

        def update(num):

            ax.clear()

            # usar el frame actual
            psi = data[num]

            plotting_psi(H, psi, node_names, ax=ax, pos=pos)

            ax.set_title(f"Time Step: {num}")

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
        ani.save( f'animation_eta_{eta}.gif', writer='pillow' )
        plt.close()

    except Exception as e:
        print(f"Error creando el gráfico: {e}")



hg_3 = {
    0: [1, 2, 3, 4, 5],
    1: [1, 6],
    2: [2, 7],
    3: [3, 8],
    4: [4, 9],
    5: [5, 10, 11,12],
    6: [6, 13, 14],
    7: [7, 15, 19, 20, 21], 
    8: [8, 16],
    9: [9, 17, 18],   
            }


alpha = 0.05
theta = 2.
eta2 = 0
eta = 1.3

matriz, nombres_nodos, nombres_aristas = fh.dict_to_matrix(hg_3)
H = hnx.Hypergraph.from_numpy_array(
    matriz, 
    node_names=nombres_nodos, 
    edge_names=nombres_aristas)

plot_animation(H, nombres_nodos, file_root=f'results_hypernet/pruebas_/net3_alpha{alpha}_theta{theta}_eta1_{2}', eta=0, scale=10)