import json
import os
from datetime import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from birb_test import BiRBTestCP
from load_account import save_account
def plotMultipleBiRBTests(results_per_percent, backend_name, qubits,file_name):
    """
    Plot the results from multiple BiRB test

    Args:
        results_per_percent (list[tuple]): List that contains tuples of the form
                                           (percent, results, valid_depths, A_fit,
                                           p_fit, mean_infidelity, mean_per_depth))

        backend_name (str): Name of the quantum processor (real or simulated)

        qubits (int): Number of qubits of the processor
            
    """
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 4))

    num_graphics = len(results_per_percent)
    colors = sns.color_palette("pastel", num_graphics)

    for i, data  in enumerate(results_per_percent): 

        # Extract all the data
        percent, results_per_depth, valid_depths, A_fit, p_fit,\
        mean_infidelity, mean_per_depth = data



        # Plot the exact points and scatter 
        parts = plt.violinplot(results_per_depth, positions=valid_depths, widths=1,
                               showmeans=True, showextrema=False, showmedians=False)

        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.3)
            pc.set_edgecolor("none")

        plt.scatter(valid_depths, mean_per_depth, color=colors[i], s=40, zorder=4)

        # Draw the curve that fit the data
        m_fit = np.linspace(min(valid_depths), max(valid_depths), 200)
        f_fit = [A_fit * p_fit ** m for m in m_fit]
        plt.plot(m_fit, f_fit, label='Percent: ' + str(percent) 
                 + ', infidelity: '+str(mean_infidelity), color=colors[i])


    plt.xlabel("Benchmark Depth")
    plt.ylabel("Polarization")

    ax = plt.gca()

    # Background
    ax.set_facecolor((0.95, 0.95, 1, 0.2)) 
    ax.set_title("Fit curve for " + backend_name + " with " 
                 + str(qubits) + " qubits")

    plt.legend(loc="upper right") 
    plt.tight_layout()
    date_now = file_name[-21:-5]
    filename = f"Fit curve for {backend_name} with {qubits} qubits {date_now}.png"
    filepath = os.path.join("Images_results", filename)
    # Save figure
    plt.savefig(filepath)
    plt.show()



def plotEvolutionPercent(percents, infidelities_per_percent,backend_name,file_name,qubits):
    """
    Plot the mean infidelities per percent of depth of a clifford circuit 

    Args:
        percents (list[float]): List of percents 

        infidelities_per_percent (list[float]): List with the infidelity for
                                                each percent
    """

    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 4))

    color = sns.color_palette("pastel", 1)[0]

    plt.plot(percents, infidelities_per_percent, label='Ideal infidelity curve', color=color, marker='o')


    plt.xlabel("Percent of a Clifford")
    plt.ylabel("Mean infidelity")

    ax = plt.gca()

    # Background
    ax.set_facecolor((0.95, 0.95, 1, 0.2)) 
    ax.set_title("Mean infidelity evolution with the percent of the clifford")

    plt.legend(loc="upper right") 
    plt.tight_layout()
    date_now = file_name[-21:-5]    # Construir el nombre del archivo
    filename = f"Mean_infidelity_evolution_with_the_percent_of_the_clifford_{backend_name}_{qubits}q_{date_now}.png"
    filepath = os.path.join("Images_results", filename)

    # Guardar figura
    plt.savefig(filepath)

    plt.show()

def read_data(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
        results_per_percent = data['results_per_percent']
        backend = data['backend_name']
        qubits = data['qubits']
        percents = [item[0] for item in results_per_percent]  # percent from each tuple

        # Obtener las mean infidelities
        infidelities_per_percent = [item[5] for item in results_per_percent]

    # Ahora puedes usarlos en tu gráfico
    plotMultipleBiRBTests(results_per_percent, backend, qubits, file_name)
    plotEvolutionPercent(percents, infidelities_per_percent, backend, file_name,qubits)

if __name__ == '__main__':
    file_name = 'experiments_results/results_fake_brisbane_3q_2025-05-20_15-28.json'
    read_data(file_name)