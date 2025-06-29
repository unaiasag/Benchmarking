import yaml
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
import re
from utils.utils import *
import warnings
import urllib3


def loadAndRunExperiments(file):
    """
    Open a file with some experiments definition and execute each of the experiments  

    Args:
        file_name (str): Name of the file where the experiments definitions are
    """

    try:
        with open(file, "r") as f:
            # Intenta leer como JSON
            data = json.load(f)
    except Exception:
        try:
            with open(file, "r") as f:
                # Si falla JSON, intenta como YAML
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
    output_folder = config["output_path"]
    output_path = Path(output_folder)

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"{output_path} must be a valid folder path.")


    # All the experiments
    experiments = data["experiments"]

    # Validate all experiments
    for exp in experiments:
        name = exp.get("name", "(no name)")
        params = exp.get("params", {})
        validateExperimentParams(params, name)
    results = {}
    # Run the experiments
    for exp in experiments:
        name = exp.get("name", "(no name)")
        params = exp.get("params", {})


        console = Console()

        table = Table(title=f"🧪 Expermient {name} details", border_style="magenta")
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Backend", params['backend'])
        table.add_row("Qubits", str(params['qubits']))
        table.add_row("Depths", str(params['depths']))
        table.add_row("Circuits per depth", str(params['circuits_per_depth']))
        table.add_row("Shots per circuit", str(params['shots_per_circuit']))
        table.add_row("Percents", str(params['percents']))

        console.print(table)


        result = runExperiment(user=user,
                      sim_type=simulation_type,
                      output_folder=output_folder,
                      backend=params['backend'],
                      qubits=params['qubits'],
                      depths=params['depths'],
                      circuits_per_depth=params['circuits_per_depth'],
                      shots_per_circuit=params['shots_per_circuit'],
                      percents=params['percents'],
                      show=False)
        results[name] = result
    # first_backend = experiments[0]["params"]["backend"] if experiments else "unknown"

    # combined_filename = f"data_combined_{first_backend}.json"
    # combined_path = Path(output_folder) / combined_filename

    # # Guardamos todos los resultados en un JSON combinado
    # with open(combined_path, "w") as f:
    #     json.dump(results, f, indent=2)
    return results



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

    plotCliffordVolume(results_per_percent,
                       backend,
                       qubits,
                       file_name,
                       show=True)

    plotMultipleBiRBTests(results_per_percent,
                          backend,
                          qubits,
                          file_name,
                          show=True)

    plotEvolutionPercent(results_per_percent,
                         backend,
                         file_name,
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

    args = parser.parse_args()

    if args.command == "run":
        results =loadAndRunExperiments(args.filepath)
        return results

    elif args.command == "show":
        readAndPlotExperiment(args.filepath)
    
def run(input_data:dict, solver_params:dict, extra_arguments:dict) -> dict:

    #
    # Add your solver's code here, or call it from here if it is already implemented in another module
    # Filtra el warning específico
    warnings.filterwarnings("ignore", category=UserWarning, module="qiskit_braket_provider.providers.adapter")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=re.escape(
            'You are running a noise-free circuit on the density matrix simulator. '
            'Consider running this circuit on the state vector simulator: LocalSimulator("default") '
            'for a better user experience.'
        )
    )
    output = main()

    # And this is the output it returns. It must be a dictionary.
    return output

if __name__ == '__main__':
    
    run(input_data={}, solver_params={}, extra_arguments={})