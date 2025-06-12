import yaml
import argparse
import json
from utils.utils import *

def loadAndRunExperiments(file):
    """
    Open a file with some experiments definition and execute each of the experiments  

    Args:
        file_name (str): Name of the file where the experiments definitions are
    """

    try:
        with open(file, "r") as f:
            experiments = yaml.safe_load(f)

    except Exception:
        print(f"[ERROR] Could not read the file :{file}")

    # Validate all the parameters of one experiment
    def validateExperimentParams(params, name="(sin nombre)"):
        if not isinstance(params.get("backend"), str):
            raise ValueError(f"[{name}] 'backend' must be an string")
        
        if not isinstance(params.get("qubits"), int):
            raise ValueError(f"[{name}] 'qubits' must be an integer")
        
        depths = params.get("depths")
        if not (isinstance(depths, list) and all(isinstance(d, int) for d in depths)):
            raise ValueError(f"[{name}] 'depths' must be a list of integers")
        
        if not isinstance(params.get("circuits_per_depth"), int):
            raise ValueError(f"[{name}] 'circuits_per_depth' must be an integer")

        if not isinstance(params.get("shots_per_circuit"), int):
            raise ValueError(f"[{name}] 'shots_per_circuit' must be an integer")
        
        percents = params.get("percents")
        if not (isinstance(percents, list) and all(isinstance(p, float) for p in percents)):
            raise ValueError(f"[{name}] 'percents' must be a list of floats")

    # Validate all experiments
    for exp in experiments:
        name = exp.get("name", "(no name)")
        params = exp.get("params", {})
        validateExperimentParams(params, name)

    # Run the experiments
    for exp in experiments:
        name = exp.get("name", "(no name)")
        params = exp.get("params", {})
        print(f"Executing: {name}")
        print(f"  Backend: {params['backend']}")
        print(f"  Qubits: {params['qubits']}")
        print(f"  Depths: {params['depths']}")
        print(f"  Circuits/depth: {params['circuits_per_depth']}")
        print(f"  Shots per circuit: {params['shots_per_circuit']}")
        print(f"  Percents: {params['percents']}")
        print()

        runExperiment(backend=params['backend'],
                      qubits=params['qubits'],
                      depths=params['depths'],
                      circuits_per_depth=params['circuits_per_depth'],
                      shots_per_circuit=params['shots_per_circuit'],
                      percents=params['percents'],
                      show=False)



def readAndPlotExperiment(file_name):
    """
    Read all the data of a experiment from a file and plot the results

    Args:
        file_name (str): Name of the file to import the experiment
    """
    file = None
    try:
        file = open(file_name,'r')
    except Exception:
        print(f"[ERROR] Could not read the file: {file_name}")

    data = json.load(file)
    backend = data['backend_name']
    qubits = data['qubits']
    results_saved = data['results_saved']
    results_per_percent = []

    # Find all the parameters for the results
    for percent_results in results_saved:
        A_fit, p_fit, mean_infidelity, mean_per_depth = fitModel(percent_results[1],
                                                                 percent_results[2],
                                                                 qubits,
                                                                 tolerance=0.5,
                                                                 initial_points=4,
                                                                 show=False)
        results_per_percent.append((
            percent_results[0],
            percent_results[1],
            percent_results[2],
            A_fit,
            p_fit,
            mean_infidelity,
            mean_per_depth
        ))

    #plotCliffordVolume(results_per_percent, backend, qubits, file_name)
    plotMultipleBiRBTests(results_per_percent, backend, qubits, file_name, show=True)

    plotEvolutionPercent(results_per_percent, backend, file_name, qubits, show=True)


def main():

    parser = argparse.ArgumentParser(description="Run a BiRB experiment,\
                                                  or show the data from \
                                                  a previous experiment")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("filepath", 
                            type=str, 
                            help="Path to the experiment definition file (.yaml)")


    show_parser = subparsers.add_parser("show", help="Show the data from a file")
    show_parser.add_argument("filepath", 
                             type=str, 
                             help="Path to the result of an experiment (.json)")

    args = parser.parse_args()

    if args.command == "run":
        loadAndRunExperiments(args.filepath)

    elif args.command == "show":
        readAndPlotExperiment(args.filepath)


if __name__ == '__main__':
    main()
