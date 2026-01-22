import yaml
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
import pickle
import numpy as np

import os
from datetime import datetime

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Session, Batch
import qiskit_ibm_runtime.fake_provider as fake_provider
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import qasm3, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import Estimator, EstimatorOptions, Sampler, SamplerOptions
from qiskit_ibm_runtime.options import TwirlingOptions, EnvironmentOptions, ResilienceOptionsV2 as ResilienceOptions

from utils.benchmark import run_GHZ_experiment, compute_l, sample_ghz_stabilizer, process_results
from utils.circuits import create_ghz_circuit

from qiskit.circuit import ParameterVector

class GHZExperiment():
    def __init__(self, num_qubits, circuit, transpiled_circuit, l, signs, m_i, observables, backend):
        """
        Docstring for __init__
        :param num_qubits: Description
        :param circuit: Description
        :param transpiled_circuit: Description
        :param l: Description
        :param observable_isa: transpiled_observable
        :param signs: Description
        :param m_i: Description
        :param observables: Description
        :param backend: Description
        """ 
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.transpiled_circuit = transpiled_circuit
        self.l = l
        self.observable_isa = observables
        self.signs = signs
        self.m_i = m_i
        self.observables = [str(obs) for obs in observables]
        self.backend = backend

    def prepare_experiment(self, execution_mode="vqc"):
        self.execution_mode = execution_mode

        if execution_mode == "vqc":

            inputs = ParameterVector("x", self.circuit.num_qubits*2)
           
            values = np.zeros(self.circuit.num_qubits*2)
            # Create fresh circuit with classical bits
            qc = QuantumCircuit(self.circuit.num_qubits, self.circuit.num_qubits)
            qc.compose(self.circuit, inplace=True)

            for qubit in range(self.circuit.num_qubits):
                gate1_id = qubit * 2
                gate2_id = gate1_id + 1

                qc.rz(inputs[gate1_id], qubit)
                qc.ry(inputs[gate2_id], qubit)
                qc.measure(qubit, qubit)

            circuit = transpile_circuit(qc, backend=self.backend)

            parameter_sets = []

            for observable in self.observables:
                iterable = iter(range(len(inputs)))

                # Add measurement in the Pauli basis
                for qubit, pauli in enumerate(observable):
                    
                    gate1_id = next(iterable)
                    gate2_id = next(iterable)

                    if pauli == 'X':
                        values[gate1_id] = np.pi
                        values[gate2_id] = np.pi/2
                    elif pauli == 'Y':
                        values[gate1_id] = np.pi/2
                        values[gate2_id] = np.pi/2
                    elif pauli in ['Z', 'I']:
                        values[gate1_id] = 0
                        values[gate2_id] = 0

                if circuit.num_parameters != len(values):
                    print(circuit.draw('text'))
                    raise ValueError(f"Number of parameters in the circuit ({circuit.num_parameters}) does not match the number of values provided ({len(values)}).")                 

                parameter_sets.append(values.tolist())
    
            self.pubs = [(circuit, parameter_sets)]

        elif execution_mode == "sampler":
            pubs = []
            for observable in self.observables:
                # Create fresh circuit with classical bits
                qc = QuantumCircuit(self.circuit.num_qubits, self.circuit.num_qubits)
                qc.compose(self.circuit, inplace=True)

                # Add measurement in the Pauli basis
                for qubit, pauli in enumerate(observable):
                    if pauli == 'X':
                        qc.rz(np.pi, qubit)
                        qc.ry(np.pi/2, qubit)
                        #qc.h(qubit)
                    elif pauli == 'Y':
                        qc.rz(np.pi/2, qubit) # S† = Rz(-π/2), but Rz(π) = Rz(-π) for the H gate, so we use Rz(π/2)
                        qc.ry(np.pi/2, qubit)
                    
                    qc.measure(qubit, qubit)

                pubs.append((qc))
            self.pubs = pubs

        elif execution_mode == "estimator":
            self.observable_isa = [observable.apply_layout(layout=self.transpiled_circuit.layout) for observable in self.observables]
            self.pubs = [(self.transpiled_circuit, observable ) for observable in self.observable_isa]
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}. Supported modes are 'estimator', 'sampler', and 'vqc'.")

    def run_experiment(self, mode):
        return run_GHZ_experiment(pubs=self.pubs, num_qubits=self.num_qubits, mode=mode, l=self.l, 
                                  observable_isa=self.observable_isa, signs=self.signs, m_i=self.m_i, 
                                  execution_mode=self.execution_mode)

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
    untranspiled_circuits = {}

    GHZ_strategies = ["lineal_v2", "lineal", "log", "log_v2"]

    for i in range(start_qubits, maximo+1):
        best_ghz_circuit = None
        best_untranspiled_circuit = None
        min_operations = float('inf')
        best_strategy = None
        for strategy in GHZ_strategies:
            untranspiled_circuit = create_ghz_circuit(i, mode=strategy)
            circuit =transpile_circuit(create_ghz_circuit(i, mode=strategy), backend=backend)
            operations_2q = circuit.count_ops().get("cz", 0) + circuit.count_ops().get("ecr", 0)
            if operations_2q < min_operations:
                min_operations = operations_2q
                best_ghz_circuit = circuit
                best_untranspiled_circuit = untranspiled_circuit
                best_strategy = strategy

        circuits[i] = best_ghz_circuit
        untranspiled_circuits[i] = best_untranspiled_circuit

    return circuits, untranspiled_circuits

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
        execution_mode = "vqc"  # Could be "estimator", "sampler", or "vqc"
        run_mode = "Session" # Could be "Session" or "Batch"
        max_retries = 3

        if params['simulacion'].lower() == "true":
            backend_name = params['backend']
            if backend_name == "None":
                backend = AerSimulator()
            else:
                FakeBackendClass = getattr(fake_provider, backend_name)
                backend_data = FakeBackendClass()

                # Create noisy simulator backend
                backend = AerSimulator().from_backend(backend_data)
        else:
            service = QiskitRuntimeService(channel="ibm_cloud", instance=params.get("instance", None), user=params.get("usuario", None))
            backend = service.backends(name=params['backend'])[0]
 
        backend_qubits = backend.configuration().n_qubits

        start_date = datetime.now()
        start_date_str = start_date.strftime("%Y%m%d_%H%M%S")

        if params['numero_maximo_qubits'] is None:
            maximo = backend_qubits
        else:
            maximo = params['numero_maximo_qubits']

        best_circuits, best_untranspiled_circuits = getBestGHZCircuitsPerQPU(backend, start_qubits=params['numero_qubits_inicial'], maximo=maximo)

        # Save the data of the experiment
        file_name_qubit_properties = f"qubit_properties_{params['backend']}_{backend_qubits}q_{start_date_str}.json"
        file_name_target = f"qubit_properties_{params['backend']}_{backend_qubits}q_{start_date_str}.pkl"
        os.makedirs(output_folder, exist_ok=True)
        filepath_qubit_properties = os.path.join(output_folder, file_name_qubit_properties)
        filepath_target = os.path.join(output_folder, file_name_target)

        # Save the circuits used in the experiment, first create a folder for them
        circuits_folder = os.path.join(output_folder, f"circuits_{params['backend']}_{start_date_str}")
        os.makedirs(circuits_folder, exist_ok=True)

        # Save the untranspiled circuits as images
        for i, untranspiled_circuit in best_untranspiled_circuits.items():
            circuit_file_image = os.path.join(circuits_folder, f"untranspiled_{i}.png")
            untranspiled_circuit.draw(output='mpl').savefig(circuit_file_image)

        qubit_properties = backend.properties()

        # Save calibration data
        if qubit_properties is not None:
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

        l = compute_l(epsilon=params['epsilon'], delta=params['delta'])  # Example values for epsilon and delta
        m_i = 1

        experiments = []
        for num_qubits in range(params['numero_qubits_inicial'], maximo + 1):

            untranspiled_circuit = best_untranspiled_circuits[num_qubits]
            # Save the circuit
            circuit = best_circuits[num_qubits]
            circuit_name = f"GHZ_{num_qubits}q"
            circuit_file = os.path.join(circuits_folder, f"{circuit_name}.pkl")
            with open(circuit_file, "wb") as f:
                pickle.dump(circuit, f)

            circuit_file_qasm = os.path.join(circuits_folder, f"{circuit_name}.qasm")
            with open(circuit_file_qasm, "w") as f:
                f.write(qasm3.dumps(circuit))

            # on test save the circuit image as well
            circuit_file_image = os.path.join(circuits_folder, f"{circuit_name}.png")
            circuit.draw(output='mpl').savefig(circuit_file_image)

            samples = sample_ghz_stabilizer(num_qubits, l)
            observables = [Pauli(p) for (p, _) in samples]
            signs = np.array([sign for (_, sign) in samples])
            
            experiment = GHZExperiment(num_qubits, untranspiled_circuit, circuit, l, signs, m_i, observables, backend)
            experiment.prepare_experiment(execution_mode=execution_mode)
            experiments.append(experiment)
        
        if run_mode == "Session":
            num_qubits = params['numero_qubits_inicial'] - 1
            backend_qubits = backend.configuration().n_qubits
            results_to_save = []
            retries = max_retries

            try:
                with Session(backend=backend) as session:
                    for experiment in experiments:
                        console.print(f"Ejecutando experimento para {num_qubits + 1} qubits...")

                        try:
                            # Run the GHZ experiment
                            result = experiment.run_experiment(mode=session)

                            observables = experiment.observables
                            num_qubits = experiment.num_qubits
                            expected_vals, fidelity_estimate = result

                            results_to_save.append(
                                (num_qubits, observables, expected_vals, fidelity_estimate)
                            )

                            if fidelity_estimate < (1 / 2 - params['epsilon']):
                                retries -= 1
                                print(
                                    f"Fidelity estimate {fidelity_estimate} is below the threshold, "
                                    f"{retries} retries left."
                                )
                            else:
                                retries = max_retries  # reset retries if successful

                            if retries <= 0:
                                print(
                                    f"Fidelity estimate {fidelity_estimate} is below the threshold, "
                                    "stopping further experiments."
                                )
                                break

                        except Exception as exp_err:
                            print(f"⚠️ Error during experiment execution: {exp_err}")
                            break  # stop further experiments but still save results

            except Exception as session_err:
                print(f"🔥 Session-level error occurred: {session_err}")

            finally:
                # ALWAYS save whatever results we have
                for results in results_to_save:
                    num_qubits, observables, expected_vals, fidelity_estimate = results

                    results_saved = {
                        "backend": params['backend'],
                        "numero_qubits_inicial": num_qubits,
                        "qubits": backend_qubits,
                        "epsilon": params['epsilon'],
                        "delta": params['delta'],
                        "observables": observables,
                        "expected_values": expected_vals.tolist(),
                        "fidelity": fidelity_estimate,
                    }

                    file_name_results = (
                        f"results_{params['backend']}_{num_qubits}q_{start_date_str}.json"
                    )
                    filepath_results = os.path.join(output_folder, file_name_results)

                    with open(filepath_results, "w") as f:
                        json.dump(results_saved, f, indent=4)

                    print("✅ Data saved in file:", filepath_results)

        
        elif run_mode == "Batch":
            batch = Batch(backend=backend)
            experiments_jobs = []
            tags = [f"{num_qubits}qb", f"l={l}"]
            twirling_options = TwirlingOptions(enable_measure = False) # To disable default measurement twirling
            environment_options = EnvironmentOptions(job_tags = tags.append(execution_mode)if tags else [execution_mode])
            resilience = ResilienceOptions(measure_mitigation = False) # To disable default measurement mitigation

            if execution_mode in ["sampler", "vqc"]:
                primitive = Sampler(mode=batch, options=SamplerOptions(default_shots=m_i, twirling=twirling_options, environment=environment_options))
            elif execution_mode == "estimator":
                primitive = Estimator(mode=batch, options=EstimatorOptions(default_shots=m_i, twirling=twirling_options, environment=environment_options, resilience=resilience, resilience_level=0))
            for experiment in experiments:
                # TODO: job partition strategy for large experiments
                jobs = [(primitive.run(experiment.pubs), experiment.observable_isa, experiment.observables, experiment.signs)]
                experiments_jobs.append(jobs)

            batch.close()

            retries = max_retries

            for experiment_jobs in experiments_jobs:
                evs = []
                all_observables = []
                
                for job, observable_isa, observables, signs in experiment_jobs:
                    evs += process_results(job.result(), execution_mode, observables=observable_isa).tolist()
                    all_observables += observables
                    
                num_qubits = len(observables[0])

                fidelity_estimate  = np.mean(np.array(evs) * np.array(signs))

                console.print(f"Procesando experimento para {num_qubits} qubits...")

                # Save results
                backend_qubits = backend.configuration().n_qubits
                
                results_saved = {
                    "backend": params['backend'],
                    "numero_qubits_inicial": num_qubits,
                    "qubits": backend_qubits,
                    "epsilon": params['epsilon'],
                    "delta": params['delta'],
                    "observables": all_observables,
                    "expected_values": evs,
                    "fidelity": fidelity_estimate,
                }
                file_name_results = f"results_{params['backend']}_{num_qubits}q_{start_date_str}.json"
                filepath_results = os.path.join(output_folder, file_name_results)

                with open(filepath_results, "w") as f:
                    json.dump(results_saved, f, indent=4)

                print("Data saved in file: " + filepath_results)

                if fidelity_estimate < (1/2 - params['epsilon']):
                    retries -=1
                    print(f"Fidelity estimate {fidelity_estimate} is below the threshold, {retries} retries left.")
                else:
                    retries = max_retries  # reset retries if successful

                if retries <=0:
                    print("Max retries reached, stopping further experiments.")
                    batch.cancel()
                    break

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
    # To be implemented: read the results from a json file and plot them
    return None

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
