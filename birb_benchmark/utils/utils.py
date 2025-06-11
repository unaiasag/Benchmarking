import os
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import lsq_linear
from birb_test import BiRBTestCP
from datetime import datetime

def plotMultipleBiRBTests(results_per_percent, 
                          backend_name, 
                          qubits, 
                          file_name, 
                          show=False):
    """
    Plot the results from multiple BiRB test

    Args:
        results_per_percent (list[tuple]): List that contains tuples of the form
                                           (percent, results, valid_depths, A_fit,
                                           p_fit, mean_infidelity, mean_per_depth))

        backend_name (str): Name of the quantum processor (real or simulated)

        qubits (int): Number of qubits of the processor

        file_name (string): Name of the file for saving figure
        
        show (bool): If true show the plots
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

    if show:
        plt.show()



def plotEvolutionPercent(percents, 
                         infidelities_per_percent, 
                         backend_name,
                         file_name, 
                         qubits, 
                         show=False):
    """
    Plot the mean infidelities per percent of depth of a clifford circuit 

    Args:
        percents (list[float]): List of percents 

        infidelities_per_percent (list[float]): List with the infidelity for
                                                each percent

        backend_name (string): For title

        file_name (string): Name of the file for saving figure

        show (bool): If true show the plots
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

    plt.legend(loc="upper left") 
    plt.tight_layout()
    date_now = file_name[-21:-5]    # Construir el nombre del archivo
    filename = f"Mean_infidelity_evolution_with_the_percent_of_the_clifford_{backend_name}_{qubits}q_{date_now}.png"
    filepath = os.path.join("images_results", filename)

    plt.savefig(filepath)

    if show:
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

    # Plot linear
    if show:
        m_fit = np.linspace(min(valid_depths), max(valid_depths), 200)
        f_fit = [logA_fit + logP_fit * m for m in m_fit]
        plt.plot(m_fit, f_fit)
        plt.scatter(log_valid_depths, log_mean_per_depth, color='red', s=40, zorder=4)
        plt.show()

    mean_infidelity = ((4**n- 1) / 4**n) * (1 - p_fit)

    return A_fit, p_fit, mean_infidelity, mean_per_depth

def runExperiment(backend, qubits, depths, circuits_per_depth, shots_per_circuit, percents, show=False):
    """
        Run an experiment and save the results in a file 

        Args:

            backend (string): Name of the IBM quantum backend (real or simulated) to run the tests on.

            qubits (int): Number of qubits available on the target quantum processor.

            depths (list[int]): List of circuit depths to be tested

            circuits_per_depth (int): Number of random circuits to generate and test for each depth

            shots_per_circuit (int): Number of shots we make for each circuit

            percents (list[float]): List of the percents of the circuit

            show (bool): If true, show the plot for each experiment
    """

    start_date_str = datetime.today().strftime('%Y-%m-%d_%H-%M')

    results_per_percent = []
    infidelities_per_percent = []

    
    for percent in percents:
        percent_str = "Percent: " + str(percent)
        print('#'*(len(percent_str) + 4))
        print('# '+percent_str + ' #')
        print('#'*(len(percent_str) + 4))


        t = BiRBTestCP(qubits, 
                       depths, 
                       "fake", 
                       backend, 
                       "david", 
                       circuits_per_depth, 
                       shots_per_circuit, 
                       percent)

        results, valid_depths = t.run()

        A_fit, p_fit, mean_infidelity, mean_per_depth = fitModel(results, valid_depths, qubits)

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
    file_name = f"results_{backend}_{qubits}q_{start_date_str}.json"
    folder = "experiments_results"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, file_name)

    saveData(results_per_percent, backend, qubits, filepath)

    plotMultipleBiRBTests(results_per_percent, backend, qubits, file_name, show)
    plotEvolutionPercent(percents, 
                         infidelities_per_percent, 
                         backend,
                         file_name, 
                         qubits,
                         show)

def plotCliffordVolume(results_per_percent, backend_name, qubits, show=False):

    clifford_volume_per_percent_results = []
    mean_clifford_volume_per_percent = []
    percents = []

    for data in results_per_percent: 

        # Extract all the data
        percent, results_per_depth, valid_depths, _, _,\
        _ , mean_per_depth = data


        if(valid_depths[0] == 1):
            percents.append(percent) 
            clifford_volume_per_percent_results.append([((4**qubits - 1) / 4**qubits) * (1 - q)  for q in results_per_depth[0]]) 
            mean_clifford_volume_per_percent.append(((4**qubits - 1) / 4**qubits) * (1 - mean_per_depth[0]))

    #print(clifford_volume_per_percent_results)

    violin_widht = 0.1
    if(len(percents) > 1):
        violin_widht = (percents[1] - percents[0])/4

    # Plot the exact points and scatter 
    parts = plt.violinplot(clifford_volume_per_percent_results, positions=percents, widths=violin_widht,
                           showmeans=True, showextrema=False, showmedians=False)


    color = sns.color_palette("pastel", 1)[0]

    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
        pc.set_edgecolor("none")

    plt.scatter(percents, mean_clifford_volume_per_percent, color=color, s=40, zorder=4)

    # Draw a line between the points
    plt.plot(percents, mean_clifford_volume_per_percent, label='Clifford volume', color=color)

    plt.xlabel("Percents")
    plt.ylabel("Infidelity")

    ax = plt.gca()

    # Background
    ax.set_facecolor((0.95, 0.95, 1, 0.2)) 
    ax.set_title("Entanglement infidelity of a clifford gate" + backend_name + " with " 
                 + str(qubits) + " qubits")

    plt.legend(loc="upper right") 
    plt.tight_layout()
    if(show):
        plt.show()
    #sns.set(style="whitegrid")
    #plt.figure(figsize=(7, 4))

    #color = sns.color_palette("pastel", 1)[0]

    #plt.plot(percents, infidelities_per_percent, label='Ideal infidelity curve', color=color, marker='o')


    #plt.xlabel("Percent of a Clifford")
    #plt.ylabel("Mean infidelity")

    #ax = plt.gca()

    ## Background
    #ax.set_facecolor((0.95, 0.95, 1, 0.2)) 
    #ax.set_title("Mean infidelity evolution with the percent of the clifford")

    #plt.legend(loc="upper right") 
    #plt.tight_layout()
    #date_now = file_name[-21:-5]    # Construir el nombre del archivo
    #filename = f"Mean_infidelity_evolution_with_the_percent_of_the_clifford_{backend_name}_{qubits}q_{date_now}.png"
    #filepath = os.path.join("images_results", filename)

    #plt.savefig(filepath)

    if show:
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
