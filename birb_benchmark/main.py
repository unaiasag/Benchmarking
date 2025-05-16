import json
import os
from datetime import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from birb_test import BiRBTestCP

def plotMultipleBiRBTests(results_per_percent, backend_name, qubits):
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
    plt.show()



def plotEvolutionPercent(percents, infidelities_per_percent):
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
    plt.show()



def saveData(results_per_percent, backend_name, qubits, file_name):
    """
    Save the data of an experiment in a json file

    Args: 
        results_per_percent (list[tuples]): Contains all the information of an eperiment.

        backend_name (str): Name of the processor who runned the experiment.

        qubits (int): Number of qubits used in the experiment.

        file_name (str): Name of the json file to save the data
    """

    data = {
        "backend_name": backend_name,
        "qubits": qubits,
        "results_per_percent": results_per_percent
    }

    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)

    print("Data saved in file: " + file_name)

def main():
    """
    Example of testing a quantum processor across different percentage configurations.
    """

    qubits = 3
    depths = [1, 2, 4, 6, 17, 36, 65, 100, 150, 220, 300]
    circuits_per_depth = 50
    backend = "fake_sherbrooke"
    percents = [0.1, 0.3, 0.5, 0.7, 1]
    start_date_str = datetime.today().strftime('%Y-%m-%d_%H-%M')

    results_per_percent = []
    infidelities_per_percent = []

    
    for percent in percents:
        print("Percent: " + str(percent))
        print("-----------")

        t = BiRBTestCP(qubits, depths, "fake", backend, "david", circuits_per_depth, percent)

        results, valid_depths = t.run()

        A_fit, p_fit, mean_infidelity, mean_per_depth = t.fitModel(results, valid_depths)

        infidelities_per_percent.append(mean_infidelity)
        results_per_percent.append((percent,
                                    results,
                                    valid_depths,
                                    A_fit, 
                                    p_fit, 
                                    mean_infidelity, 
                                    mean_per_depth))

        print()


        
    # Save the data of the experiment
    filename = f"results_{backend}_{qubits}q_{start_date_str}.json"
    folder = "experiments_results"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    saveData(results_per_percent, backend, qubits, filepath)

    plotMultipleBiRBTests(results_per_percent, backend, qubits)
    plotEvolutionPercent(percents, infidelities_per_percent)


if __name__ == '__main__':
    main()
