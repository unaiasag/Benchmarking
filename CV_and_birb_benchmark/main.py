import yaml
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from datetime import datetime

from utils.utils import *

def loadAndRunExperiments(file):
    """
    Open a file with some experiments definition and execute each of the experiments  

    Args:
        file_name (str): Name of the file where the experiments definitions are

        circuits_folder (str): Name of the folder containing the transpiled circuits
                               for 'real' execution
    """

    try:
        with open(file, "r") as f:
            data = yaml.safe_load(f)

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


    # General config 
    config = data["config"]
    user = config["user"]
    simulation_type = config["simulation_type"]

    #if simulation_type == "real":
    #    if not circuits_folder:
    #        raise ValueError("[ERROR] For 'real' executions, you must specify --circuits")
    #    circuits_path = Path(circuits_folder)
    #    if not circuits_path.exists():
    #        raise FileNotFoundError(f"[ERROR] Provided circuits folder '{circuits_folder}' does not exist.")
    #else:
    #    circuits_path = None


    execution_mode = config.get("execution_mode")
    if execution_mode == None:
        execution_mode = "job"
        print(f"Default execution mode: 'job'")

    output_folder = config["output_path"]
    output_path = Path(output_folder)

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"{output_path} must be a valid folder path.")

    circuits_folder = config["circuits_path"]
    circuits_path = Path(circuits_folder)

    if not circuits_path.exists() or not circuits_path.is_dir():
        raise ValueError(f"{output_path} must be a valid folder path.")

    # All the experiments
    experiments = data["experiments"]

    # Validate all experiments
    for exp in experiments:
        name = exp.get("name", "(no name)")
        params = exp.get("params", {})
        validateExperimentParams(params, name)

    # Run the experiments
    count = 1
    for exp in experiments:
        name = exp.get("name")
        if not name:
            name = f"unnamed_{count}"
            count += 1
        params = exp.get("params", {})


        console = Console()

        table = Table(title=f"🧪 Expermient {name} detalis", border_style="magenta")
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Backend", params['backend'])
        table.add_row("Qubits", str(params['qubits']))
        table.add_row("Depths", str(params['depths']))
        table.add_row("Circuits per depth", str(params['circuits_per_depth']))
        table.add_row("Shots per circuit", str(params['shots_per_circuit']))
        table.add_row("Percents", str(params['percents']))

        console.print(table)

        experiment_path = os.path.join(circuits_path, name)


        runExperiment(user=user,
                      sim_type=simulation_type,
                      execution_mode=execution_mode,
                      circuits_folder=experiment_path,
                      output_folder=output_folder,
                      backend=params['backend'],
                      qubits=params['qubits'],
                      depths=params['depths'],
                      circuits_per_depth=params['circuits_per_depth'],
                      shots_per_circuit=params['shots_per_circuit'],
                      percents=params['percents'],
                      show=False)

def loadAndPrepareExperiments(file):
    """
    Open a file with some experiments definition and transpile the corresponding circuits  

    Args:
        file_name (str): Name of the file where the experiments definitions are
    """

    try:
        with open(file, "r") as f:
            data = yaml.safe_load(f)

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


    # General config 
    config = data["config"]
    user = config["user"]
    simulation_type = config["simulation_type"]

    execution_mode = config.get("execution_mode")
    if execution_mode == None:
        execution_mode = "job"
        print(f"Default execution mode: 'job'")

    output_folder = config["output_path"]
    output_path = Path(output_folder)

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"{output_path} must be a valid folder path.")

    circuits_folder = config["circuits_path"]
    circuits_path = Path(circuits_folder)

    if not circuits_path.exists() or not circuits_path.is_dir():
        raise ValueError(f"{output_path} must be a valid folder path.")

    # All the experiments
    experiments = data["experiments"]

    # Validate all experiments
    for exp in experiments:
        name = exp.get("name", "(no name)")
        params = exp.get("params", {})
        validateExperimentParams(params, name)

    #filename = os.path.basename(file)
    #folder_name = os.path.splitext(filename)[0] + datetime.today().strftime('_%Y-%m-%d_%H-%M')
    #circuits_folder = os.path.join("transpiled_circuits", folder_name)

    # Prepare the experiments
    count = 1
    for exp in experiments:
        name = exp.get("name")
        if not name:
            name = f"unnamed_{count}"
            count += 1
        params = exp.get("params", {})


        console = Console()

        table = Table(title=f"🧪 Expermient {name} detalis", border_style="magenta")
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Backend", params['backend'])
        table.add_row("Qubits", str(params['qubits']))
        table.add_row("Depths", str(params['depths']))
        table.add_row("Circuits per depth", str(params['circuits_per_depth']))
        table.add_row("Shots per circuit", str(params['shots_per_circuit']))
        table.add_row("Percents", str(params['percents']))

        console.print(table)

        experiment_path = os.path.join(circuits_folder, name)
        os.makedirs(experiment_path, exist_ok=True)

        prepareExperiment(user=user,
                          sim_type=simulation_type,
                          execution_mode=execution_mode,
                          output_folder=experiment_path,
                          backend=params['backend'],
                          qubits=params['qubits'],
                          depths=params['depths'],
                          circuits_per_depth=params['circuits_per_depth'],
                          shots_per_circuit=params['shots_per_circuit'],
                          percents=params['percents'])

def readAndPlotExperiment(file, datetime):
    """
    Read all the data of a experiment from a file and plot the results

    Args:
        file_name (str): Name of the file to import the experiment
    """
    #file = None
    #try:
    #    file = open(file_name,'r')
    #except Exception:
    #    print(f"[ERROR] Could not read the file: {file_name}")

    try:
        with open(file, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        print(f"[ERROR] Could not read the file :{file}")

    backend = data['experiments'][0]['params']['backend']#['backend_name']
    qubits = data['experiments'][0]['params']['qubits']#data['qubits']
    results_file_name = data['config']['output_path'] + '/' + 'results_' + backend + '_' + str(qubits) + 'q_' + datetime + '.json'

    try:
        with open(results_file_name, "r") as f:
            results_object = json.load(f)
    except Exception:
        print(f"[ERROR] Could not read the file :{results_file_name}")

    results_per_percent = []
    results_saved = results_object['results_saved']

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

    plotCliffordVolume(results_per_percent,
                       backend,
                       qubits,
                       file,
                       show=True)

    plotMultipleBiRBTests(results_per_percent,
                          backend,
                          qubits,
                          file,
                          show=True)

    plotEvolutionPercent(results_per_percent,
                         backend,
                         file,
                         qubits,
                         show=True)

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
    show_parser.add_argument("datetime", 
                             type=str, 
                             help="Date and time of the execution")
    

    transpile_parser = subparsers.add_parser("transpile", help="Transpile circuits of an experiment")
    transpile_parser.add_argument("filepath", 
                                  type=str, 
                                  help="Path to the experiment definition file (.yaml)")

    args = parser.parse_args()

    if args.command == "run":
        loadAndRunExperiments(args.filepath)

    elif args.command == "show":
        readAndPlotExperiment(args.filepath, args.datetime)
    
    elif args.command == "transpile":
        loadAndPrepareExperiments(args.filepath)


if __name__ == '__main__':
    main()
