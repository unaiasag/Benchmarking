import argparse
import json
import os
from datetime import datetime

from utils.utils import *

from birb_test import BiRBTestCP

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



def runExperiment():
    """
    Example of testing a quantum processor across different percentage configurations. 
    """

    qubits = 3
    backend = "fake_sherbrooke"
    depths = [1, 2, 4, 6, 17, 36, 65, 100, 150, 220, 300]
    circuits_per_depth = 50
    percents = [0.1, 0.3, 0.5, 0.7, 1]

    start_date_str = datetime.today().strftime('%Y-%m-%d_%H-%M')

    results_per_percent = []
    infidelities_per_percent = []

    
    for percent in percents:
        print("Percent: " + str(percent))
        print("-----------")

        t = BiRBTestCP(qubits, depths, "fake", backend, "david", circuits_per_depth, percent)

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

    plotMultipleBiRBTests(results_per_percent, backend, qubits, file_name)
    plotEvolutionPercent(percents, infidelities_per_percent, backend, file_name, qubits)

def readAndPlotExperiment(file_name):
    """
    Read all the data of a experiment from a file and plot the results

    Args:
        file_name (str): Name of the file to import the experiment
    """
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
            results_per_percent = data['results_per_percent']
            backend = data['backend_name']
            qubits = data['qubits']
            percents = [item[0] for item in results_per_percent]  
            infidelities_per_percent = [item[5] for item in results_per_percent]

            plotMultipleBiRBTests(results_per_percent, backend, qubits, file_name)
            plotEvolutionPercent(percents, 
                                 infidelities_per_percent, 
                                 backend,
                                 file_name, 
                                 qubits)

    except Exception:
        print(f"[ERROR] Could not read the file :{file_name}")

def main():

    parser = argparse.ArgumentParser(description="Percent BiRB tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomand run
    run_parser = subparsers.add_parser("run", help="Run an experiment")

    # TODO: Add a file with the parameters of the experiment

    # Subcomand: read
    show_parser = subparsers.add_parser("read", help="Show the data from a file")
    show_parser.add_argument("filepath", type=str, help="Path to the file")

    args = parser.parse_args()

    if args.command == "run":
        runExperiment()

    elif args.command == "read":
        readAndPlotExperiment(args.filepath)


if __name__ == '__main__':
    main()
