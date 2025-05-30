import json
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import lsq_linear

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

    # Save figure
    date_now = file_name[-21:-5]
    filename = f"Fit curve for {backend_name} with {qubits} qubits {date_now}.png"
    filepath = os.path.join("images_results", filename)

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
    filepath = os.path.join("images_results", filename)

    # Guardar figura
    plt.savefig(filepath)

    plt.show()



def fitModel(results_per_depth, valid_depths, n, tolerance=0.5, initial_points=3, show=False):

    """
    Given the results of a test, fits an exponential model to the data.

    Args:
        results_per_depth (list[list[float]]): List of results for each
                                               executed depth.

        valid_depths (list[int]): List of depths corresponding to each set
                                  of results.

        n (int): Number of qubits.

        tolerance (float): Allowed distance from one point to the expected line
        in linear regression 

        initial_points (int): Number of points to start making linear regression

        show (bool): If true, plot the regression

    Returns:
        A_fit (float): Estimated SPAM (State Preparation and Measurement)
                       error of the model.

        p_fit (float): Estimated polarization per layer.

        mean_infidelity (float): Estimated average infidelity per layer.
        
        mean_per_depth (list[float]): Mean of the results for each depth.
    """

    log_mean_per_depth = []
    mean_per_depth = []
    log_valid_depths = []
    for i, depth_results in enumerate(results_per_depth):
        mean = statistics.mean(depth_results)
        mean_per_depth.append(mean)

        # We skip low values
        if(mean > 0):
            log_mean_per_depth.append(np.log(mean))
            log_valid_depths.append(valid_depths[i])



    # Linear regresion for k first points
    def regression(k):
        A = np.vstack([log_valid_depths[0:k], np.ones_like(log_valid_depths[0:k])]).T 
        res = lsq_linear(A, log_mean_per_depth[0:k], bounds=([-np.inf, -np.inf], [0, 0])) 
        logP_fit, logA_fit = res.x
        return logP_fit, logA_fit

    logP_fit, logA_fit = regression(initial_points)

    # Check if the next point is so far from the predicted line
    k = initial_points
    for point in range(initial_points, len(log_mean_per_depth)):
        expected_y = logP_fit * log_valid_depths[point] + logA_fit
        real_y = log_mean_per_depth[point]

        if(abs(real_y - expected_y) > tolerance):
            break

        k += 1
        logP_fit, logA_fit = regression(k)




    A_fit = np.exp(logA_fit)
    p_fit = np.exp(logP_fit)

    if show:
        # Plot linear
        m_fit = np.linspace(min(valid_depths), max(valid_depths), 200)
        f_fit = [logA_fit + logP_fit * m for m in m_fit]
        plt.plot(m_fit, f_fit)
        plt.scatter(log_valid_depths, log_mean_per_depth, color='red', s=40, zorder=4)
        plt.show()

    mean_infidelity = ((4**n- 1) / 4**n) * (1 - p_fit)

    return A_fit, p_fit, mean_infidelity, mean_per_depth

def readData(file_name):
    """
    Read all the data of a experiment from a file and plot the results

    Args:
        file_name (str): Name of the file to import the experiment
    """
    with open(file_name, 'r') as file:
        data = json.load(file)
        results_per_percent = data['results_per_percent']
        backend = data['backend_name']
        qubits = data['qubits']

        # Obtain mean infidelities
        for item in results_per_percent:
            A_fit, p_fit, mean_infidelity, _ = fitModel(item[1], item[2], qubits ,show=True)
            item[3] = A_fit
            item[4] = p_fit
            item[5] = mean_infidelity

        percents = [item[0] for item in results_per_percent]  
        infidelities_per_percent = [item[5] for item in results_per_percent]

    # Ahora puedes usarlos en tu gráfico
    plotMultipleBiRBTests(results_per_percent, backend, qubits, file_name)
    plotEvolutionPercent(percents, infidelities_per_percent, backend, file_name, qubits)

if __name__ == '__main__':
    file_name = 'experiments_results/Fake_experiments/results_fake_toronto_7q_2025-05-21_16-55.json'
    readData(file_name)
