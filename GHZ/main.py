import yaml
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
import pickle

import os
from datetime import datetime

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Session
import qiskit_ibm_runtime.fake_provider as fake_provider

from utils.benchmark import run_GHZ_experiment
from utils.circuits import create_ghz_circuit


def transpile_circuit(circuit, backend, optimization_level=3):
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    transpiled_circuit = pass_manager.run(circuit)
    return transpiled_circuit

def getBestGHZCircuitsPerQPU(backend, start_qubits=2, maximo=2):
    """
    Transpila y selecciona los mejores circuitos según la calibración del backend.

    Args:
        backend_name (str): Nombre del backend (ej. 'ibmq_belem')
        qubits (int): Número de qubits
        calibrations (dict): Calibraciones del backend
        limit (int): Número máximo de circuitos a seleccionar

    Returns:
        List[QuantumCircuit]: Circuitos transpileados optimizados
    """
    backend.configuration().n_qubits
    qubits = backend.configuration().n_qubits

    circuits = {}

    for i in range(start_qubits, maximo+1):
        # TODO: Implementar la lógica para obtener el circuito óptimo
        best_mode = "lineal_v2"  # Placeholder for the best mode, best mode for transpiled circuits measuring 2 qubit gates
        circuits[i] = transpile_circuit(create_ghz_circuit(i, mode=best_mode), backend=backend)
        

    return circuits

def validateExperimentParams(params, name="(sin nombre)"):
    if not isinstance(params.get("backend"), str):
        raise ValueError(f"[{name}] 'backend' debe ser un string")
    if not isinstance(params.get("usuario"), str):
        raise ValueError(f"[{name}] 'usuario' debe ser un string")
    if not isinstance(params.get("delta"), float):
        raise ValueError(f"[{name}] 'delta' debe ser un float")
    if not isinstance(params.get("epsilon"), float):
        raise ValueError(f"[{name}] 'epsilon' debe ser un float")
    if not isinstance(params.get("numero_qubits_inicial"), int):
        raise ValueError(f"[{name}] 'numero_qubits_inicial' debe ser un entero")
    if params.get("numero_qubits_inicial", 0) < 2:
        raise ValueError(f"[{name}] 'numero_qubits_inicial' debe ser al menos 2")
    if not isinstance(params.get("simulacion"), str):
        raise ValueError(f"[{name}] 'simulacion' debe ser un string")
    if not isinstance(params.get("numero_maximo_qubits"), int):
        print(params.get("numero_maximo_qubits"))
        raise ValueError(f"[{name}] 'numero_maximo_qubits' debe ser un entero")
    if params.get("numero_maximo_qubits", 0) < 2:
        raise ValueError(f"[{name}] 'numero_maximo_qubits' debe ser al menos 2")
    if params.get("simulacion", "").lower() not in ["true", "false"]:
        raise ValueError(f"[{name}] 'simulacion' debe ser 'true' o 'false'")

def loadAndRunExperiments(file):
    try:
        with open(file, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        print(f"[ERROR] No se pudo leer el archivo: {file}")
        return

    config = data.get("config", {})
    output_folder = config.get("output_path", "resultados")
    output_path = Path(output_folder)

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"{output_path} no es un directorio válido")

    experiments = data["experiments"]

    for exp in experiments:
        name = exp.get("name", "(sin nombre)")
        params = exp.get("params", {})
        validateExperimentParams(params, name)

    for exp in experiments:
        name = exp.get("name", "(sin nombre)")
        params = exp.get("params", {})

        console = Console()
        table = Table(title=f"🧪 Detalles del experimento: {name}", border_style="magenta")
        table.add_column("Parámetro", style="bold cyan")
        table.add_column("Valor", style="white")

        table.add_row("Backend", params['backend'])
        table.add_row("Usuario", params['usuario'])
        table.add_row("Delta", str(params['delta']))
        table.add_row("epsilon", str(params['epsilon']))
        table.add_row("Qubits Iniciales", str(params['numero_qubits_inicial']))
        table.add_row("Simulación", params['simulacion'])

        console.print(table)

        if params['simulacion'].lower() == "true":
            FakeBackendClass = getattr(fake_provider, params['backend'])
            backend_data = FakeBackendClass()

            # Create noisy simulator backend
            backend = AerSimulator().from_backend(backend_data)
        else:
            provider = IBMQ.load_account()
            backend = provider.get_backend(params['backend'])
 
        backend_qubits = backend.configuration().n_qubits

        start_date = datetime.now()
        start_date_str = start_date.strftime("%Y%m%d_%H%M%S")

        if params['numero_maximo_qubits'] is None:
            maximo = backend_qubits
        else:
            maximo = params['numero_maximo_qubits']

        best_circuits = getBestGHZCircuitsPerQPU(backend, start_qubits = params['numero_qubits_inicial'], maximo=maximo)    
        # Save the data of the experiment
 
        file_name_qubit_properties = f"qubit_properties_{params['backend']}_{backend_qubits}q_{start_date_str}.json"
        file_name_target = f"qubit_properties_{params['backend']}_{backend_qubits}q_{start_date_str}.pkl"
        os.makedirs(output_folder, exist_ok=True)
        filepath_qubit_properties = os.path.join(output_folder, file_name_qubit_properties)
        filepath_target = os.path.join(output_folder, file_name_target)

        qubit_properties = backend.properties()

    
        with Session(backend=backend) as session:
            for num_qubits in range(params['numero_qubits_inicial'], maximo + 1):
                console.print(f"Ejecutando experimento para {num_qubits} qubits...")

                # Run the GHZ experiment (returns observables, expected values, and fidelity estimate)
                result = run_GHZ_experiment(
                    num_qubits=num_qubits,
                    mode=session,
                    transpiled_circuit=best_circuits[num_qubits],
                    epsilon=params['epsilon'],
                    delta=params['delta']
                )

                # Save results
                observables, expected_vals, fidelity_estimate = result
                results_saved = {
                    "backend": params['backend'],
                    "numero_qubits_inicial": num_qubits,
                    "qubits": backend_qubits,
                    "epsilon": params['epsilon'],
                    "delta": params['delta'],
                    "observables": [str(obs) for obs in observables],
                    "expected_values": expected_vals.tolist(),
                    "fidelity": fidelity_estimate,
                }
                file_name_results = f"results_{params['backend']}_{num_qubits}q_{start_date_str}.json"
                filepath_results = os.path.join(output_folder, file_name_results)

                with open(filepath_results, "w") as f:
                    json.dump(results_saved, f, indent=4)

                print("Data saved in file: " + filepath_results)

                if fidelity_estimate < (1/2 - params['epsilon']):
                    print(f"Fidelity estimate {fidelity_estimate} is below the threshold, stopping further experiments.")
                    break

        # Save calibration data
        qubit_properties_list = []
        for i in range(backend_qubits):
            qubit_property_dict = {}
            qubit_property_dict["number"] = i
            qubit_property_dict["T1"] = qubit_properties.qubit_property(i).get("T1", [None])[0]
            qubit_property_dict["T2"] = qubit_properties.qubit_property(i).get("T2", [None])[0]
            qubit_property_dict["frequency"] = qubit_properties.qubit_property(i).get("frequency", [None])[0]
            qubit_property_dict["anharmonicity"] = qubit_properties.qubit_property(i).get("anharmonicity", [None])[0]
            qubit_property_dict["readout_error"] = qubit_properties.qubit_property(i).get("readout_error", [None])[0]
            qubit_property_dict["prob_meas0_prep1"] = qubit_properties.qubit_property(i).get("prob_meas0_prep1", [None])[0]
            qubit_property_dict["prob_meas1_prep0"] = qubit_properties.qubit_property(i).get("prob_meas1_prep0", [None])[0]
            qubit_property_dict["readout_length"] = qubit_properties.qubit_property(i).get("readout_length", [None])[0]
            qubit_properties_list.append(qubit_property_dict)
        target = backend.target
        
        saveCalibration(params['backend'],
                        backend_qubits,
                        qubit_properties_list,
                        target,
                        filepath_qubit_properties,
                        filepath_target)


def saveCalibration(backend_name, 
                    qubits, 
                    qubit_properties_list,
                    target,
                    filename_qubit_properties,
                    filename_target):
    """
    Save the data of an experiment in a json file
    Args: 
        backend_name (str): Name of the processor who runned the experiment.
        qubits (int): Number of qubits used in the experiment.
        qubit_properties_list (list): List of dictionaries containing the calibration of each qubit.
        target (object): qiskit object containing the calibration of connections among qubits.
    """
    data = {
        "backend_name": backend_name,
        "qubits": qubits,
        "qubit_properties_list": qubit_properties_list,
    }

    with open(filename_qubit_properties, "w") as f1:
        json.dump(data, f1, indent=4)
    with open(filename_target, "wb") as f2:
        pickle.dump(target, f2)

    print("Calibration data saved in files: " + filename_qubit_properties + " and " + filename_target)

def readAndPlotExperiment(file_name):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
    except Exception:
        print(f"[ERROR] No se pudo leer el archivo: {file_name}")
        return

    backend = data.get('backend', 'desconocido')
    qubits = data.get('numero_qubits_inicial', 0)
    results_saved = data.get('results_saved', [])
    results_per_percent = []

    for percent_results in results_saved:
        A_fit, p_fit, mean_infidelity, mean_per_depth = fitModel(
            percent_results[1],
            percent_results[2],
            qubits,
            tolerance=0.5,
            initial_points=4,
            show=False
        )
        results_per_percent.append((
            percent_results[0],
            percent_results[1],
            percent_results[2],
            A_fit,
            p_fit,
            mean_infidelity,
            mean_per_depth
        ))

def main():
    parser = argparse.ArgumentParser(description="Ejecuta un benchmark BiRB y muestra resultados previos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Ejecutar un experimento")
    run_parser.add_argument("filepath", type=str, help="Ruta al archivo .yml de definición del experimento")

    show_parser = subparsers.add_parser("show", help="Mostrar resultados desde un archivo .json")
    show_parser.add_argument("filepath", type=str, help="Ruta al archivo .json del experimento")

    args = parser.parse_args()

    if args.command == "run":
        loadAndRunExperiments(args.filepath)
    elif args.command == "show":
        readAndPlotExperiment(args.filepath)

if __name__ == '__main__':
    main()
