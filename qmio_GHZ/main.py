#import yaml
import argparse
import json
from pathlib import Path
import pickle
import numpy as np

import os
from datetime import datetime

from qiskit.transpiler import generate_preset_pass_manager
from qiskit import qasm3, QuantumCircuit
from qiskit.quantum_info import Pauli

from utils.benchmark import run_GHZ_experiment, compute_l, sample_ghz_stabilizer
from utils.circuits import create_ghz_circuit

from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qmiotools.integrations.qiskitqmio import QmioBackend
from qmiotools.integrations.qiskitqmio import FakeQmio

class GHZExperiment():
    def __init__(self, num_qubits, circuit, transpiled_circuit, l, signs, m_i, observables, backend, benchmark="RS"):
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
        :param benchmark: Description
        """ 
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.transpiled_circuit = transpiled_circuit
        self.l = l
        self.observable_isa = observables
        self.signs = signs
        self.m_i = m_i
        self.observables = [str(obs) for obs in observables] if observables is not None else None
        self.backend = backend
        self.benchmark = benchmark

    def prepare_experiment(self, execution_mode="vqc"):
        self.execution_mode = execution_mode

        if self.benchmark == "RS":

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
        
        elif self.benchmark == "Witnesses":
            if execution_mode == "vqc":

                N = self.circuit.num_qubits
                inputs = ParameterVector("x", N * 2)
                values = np.zeros(N * 2)

                qc = QuantumCircuit(N, N)
                qc.compose(self.circuit, inplace=True)

                for qubit in range(N):
                    gate1_id = qubit * 2       
                    gate2_id = gate1_id + 1    

                    qc.rz(inputs[gate1_id], qubit)
                    qc.ry(inputs[gate2_id], qubit)
                    qc.measure(qubit, qubit)

                circuit = transpile_circuit(qc, backend=self.backend)

                parameter_sets = []

                # POPULATION 
                values = np.zeros(N * 2)

                for qubit in range(N):
                    gate1_id = qubit * 2
                    gate2_id = gate1_id + 1

                    values[gate1_id] = 0.0     # Rz
                    values[gate2_id] = 0.0     # Ry

                parameter_sets.append(values.tolist())

                # COHERENCE
                for k in range(1, N + 1):

                    theta_k = k * np.pi / N
                    values = np.zeros(N * 2)

                    for qubit in range(N):
                        gate1_id = qubit * 2
                        gate2_id = gate1_id + 1

                        values[gate1_id] = -theta_k     # Rz(-θk)
                        values[gate2_id] = -np.pi / 2   # Ry(-π/2)

                    parameter_sets.append(values.tolist())

                # sanity check
                if circuit.num_parameters != len(values):
                    print(circuit.draw('text'))
                    raise ValueError(
                        f"Number of parameters ({circuit.num_parameters}) "
                        f"!= provided ({len(values)})"
                    )

                self.pubs = [(circuit, parameter_sets)]
            else:
                raise ValueError(f"Unsupported execution mode for Witnesses benchmark: {execution_mode}. Supported mode is 'vqc'.")
        else:
            raise ValueError(f"Unsupported benchmark type: {self.benchmark}. Supported benchmarks are 'RS' and 'Witnesses'.")

    def run_experiment(self):
        if self.benchmark == "RS":
            return run_GHZ_experiment(backend=self.backend, pubs=self.pubs, num_qubits=self.num_qubits, l=self.l, 
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

def loadAndRunExperiments():
    

    output_folder = "experiment_results"
    output_path = Path(output_folder)

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"{output_path} no es un directorio válido")

    
    name = "GHZ_Benchmark_Test"
    backend_name= "qmio"
    delta= 0.1                                  # Confidence level parameter (float)
    epsilon= 0.1                                # Accuracy level parameter (float)
    numero_qubits_inicial= 2                     # Minimum number of qubits to start from (int, at least 2)
    numero_maximo_qubits= 4                # Maximum number of qubits to test (int, at least 2)
    simulacion= "false"


    execution_mode = "vqc"  # Could be "estimator", "sampler", or "vqc"
    search_strategy = "binary"  # Could be "linear" or "binary"
    benchmark = "RS" # Could be "RS" or "Witnesses"
    max_retries = 3

    if simulacion.lower() == "true":
        # aer simulator backend
        backend = AerSimulator()
        #backend = FakeQmio()
    else:
        backend=QmioBackend()

    backend_qubits = 32

    start_date = datetime.now()
    start_date_str = start_date.strftime("%Y%m%d_%H%M%S")

    if numero_maximo_qubits is None:
        maximo = backend_qubits
    else:
        maximo = numero_maximo_qubits

    best_circuits, best_untranspiled_circuits = getBestGHZCircuitsPerQPU(backend, start_qubits=numero_qubits_inicial, maximo=maximo)

    # Save the data of the experiment
    file_name_qubit_properties = f"qubit_properties_{backend_name}_{backend_qubits}q_{start_date_str}.json"
    file_name_target = f"qubit_properties_{backend_name}_{backend_qubits}q_{start_date_str}.pkl"
    os.makedirs(output_folder, exist_ok=True)
    filepath_qubit_properties = os.path.join(output_folder, file_name_qubit_properties)
    filepath_target = os.path.join(output_folder, file_name_target)

    # Save the circuits used in the experiment, first create a folder for them
    circuits_folder = os.path.join(output_folder, f"circuits_{backend_name}_{start_date_str}")
    os.makedirs(circuits_folder, exist_ok=True)

    # Save the untranspiled circuits as images
    for i, untranspiled_circuit in best_untranspiled_circuits.items():
        circuit_file_image = os.path.join(circuits_folder, f"untranspiled_{i}.png")
        untranspiled_circuit.draw(output='mpl').savefig(circuit_file_image)

    qubit_properties_list = extract_qubit_properties(backend, backend_qubits)
    target = getattr(backend, "target", None)

    saveCalibration(
        backend_name,
        backend_qubits,
        qubit_properties_list,
        target,
        filepath_qubit_properties,
        filepath_target
    )
    
    l = compute_l(epsilon=epsilon, delta=delta)  # Example values for epsilon and delta
    m_i = 1

    experiments = {}
    smaller_size = numero_qubits_inicial
    for num_qubits in range(smaller_size, maximo + 1):

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

        if benchmark == "RS":
            samples = sample_ghz_stabilizer(num_qubits, l)
            observables = [Pauli(p) for (p, _) in samples]
            signs = np.array([sign for (_, sign) in samples])
            experiment = GHZExperiment(num_qubits, untranspiled_circuit, circuit, l, signs, m_i, observables, backend, benchmark=benchmark)
        experiment.prepare_experiment(execution_mode=execution_mode)
        experiments[num_qubits] = experiment
    
    
    num_qubits = numero_qubits_inicial - 1
    backend_qubits = 32
    results_to_save = []
    retries = max_retries

    try:
        
        if search_strategy == "linear":
            for num_qubits, experiment in experiments.items():
                try:
                    # Run the GHZ experiment
                    result = experiment.run_experiment()

                    observables = experiment.observables
                    num_qubits = experiment.num_qubits
                    expected_vals, fidelity_estimate = result

                    results_to_save.append(
                        (num_qubits, observables, expected_vals, fidelity_estimate)
                    )

                    if fidelity_estimate < (1 / 2 - epsilon):
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

        elif search_strategy == "binary":
            max_retries = 0  # no retries in binary search, as we will discard half of the search space after each experiment
            max_idx = maximo
            min_idx = smaller_size
            idx = int((max_idx+min_idx) // 2)
            end = False

            while not end:
                experiment = experiments[idx]
                try:
                    # Run the GHZ experiment
                    result = experiment.run_experiment()

                    observables = experiment.observables
                    num_qubits = experiment.num_qubits
                    if benchmark == "RS":
                        expected_vals, fidelity_estimate = result

                        results_to_save.append(
                            (num_qubits, observables, expected_vals, fidelity_estimate)
                        )
                    else: # Witnesses benchmark
                        all_counts, P, C, fidelity_estimate = result

                        results_to_save.append(
                            (num_qubits, all_counts, {"P": P, "C": C}, fidelity_estimate)
                        )

                    if fidelity_estimate < (1 / 2 - epsilon):
                        # If the fidelity is below the threshold discard the upper half of the search space
                        max_idx = idx - 1
                        if min_idx > max_idx:
                            print(
                                f"Fidelity estimate {fidelity_estimate} crosses threshold at {idx} qubits. Stopping further experiments."
                            )
                            end = True
                        else:
                            idx = int((max_idx+min_idx) // 2)
                    else:
                        # If the fidelity is above the threshold discard the lower half of the search space
                        min_idx = idx + 1
                        if min_idx > max_idx:
                            print(
                                f"Fidelity estimate {fidelity_estimate} crosses threshold at {idx+1} qubits. Stopping further experiments."
                            )
                            end = True
                        else:
                            idx = int((max_idx+min_idx) // 2)


                except Exception as exp_err:
                    print(f"⚠️ Error during experiment execution: {exp_err}")
                    end = True  # stop further experiments but still save results

    except Exception as session_err:
        print(f"🔥 Session-level error occurred: {session_err}")

    finally:
        # ALWAYS save whatever results we have
        for results in results_to_save:
            if benchmark == "RS":
                num_qubits, observables, expected_vals, fidelity_estimate = results

                results_saved = {
                    "backend": backend_name,
                    "numero_qubits_inicial": num_qubits,
                    "qubits": backend_qubits,
                    "epsilon": epsilon,
                    "delta": delta,
                    "observables": observables,
                    "expected_values": expected_vals.tolist(),
                    "fidelity": fidelity_estimate,
                }
            else: # Witnesses benchmark
                num_qubits, all_counts, P_C_dict, fidelity_estimate = results

                results_saved = {
                    "backend": backend_name,
                    "numero_qubits_inicial": num_qubits,
                    "qubits": backend_qubits,
                    "epsilon": epsilon,
                    "delta": delta,
                    "all_counts": all_counts,
                    "P_C": P_C_dict,
                    "fidelity": fidelity_estimate,
                }

            file_name_results = (
                f"results_{backend_name}_{num_qubits}q_{start_date_str}.json"
            )
            filepath_results = os.path.join(output_folder, file_name_results)

            with open(filepath_results, "w") as f:
                json.dump(results_saved, f, indent=4)

            print("✅ Data saved in file:", filepath_results)

def extract_qubit_properties(backend, backend_qubits):
    qubit_properties_list = []
    
    # Comprobar si el backend usa la interfaz legacy BackendV1
    has_v1_properties = hasattr(backend, "properties") and callable(backend.properties) and backend.properties() is not None

    if has_v1_properties:
        # --- Extracción BackendV1 (IBM Legacy) ---
        qubit_props = backend.properties()
        for i in range(backend_qubits):
            q_dict = {
                "number": i,
                "T1": qubit_props.qubit_property(i).get("T1", [None])[0],
                "T2": qubit_props.qubit_property(i).get("T2", [None])[0],
                "frequency": qubit_props.qubit_property(i).get("frequency", [None])[0],
                "anhharmonicity": qubit_props.qubit_property(i).get("anharmonicity", [None])[0],
                "readout_error": qubit_props.qubit_property(i).get("readout_error", [None])[0],
                "prob_meas0_prep1": qubit_props.qubit_property(i).get("prob_meas0_prep1", [None])[0],
                "prob_meas1_prep0": qubit_props.qubit_property(i).get("prob_meas1_prep0", [None])[0],
                "readout_length": qubit_props.qubit_property(i).get("readout_length", [None])[0],
            }
            qubit_properties_list.append(q_dict)
            
    else:
        # --- Extracción BackendV2 (QMIO / IBM V2) ---
        target = getattr(backend, "target", None)
        
        for i in range(backend_qubits):
            q_dict = {"number": i}
            
            # 1. T1, T2, Frecuencia y Anarmonicidad desde qubit_properties
            qprop = None
            if hasattr(backend, "qubit_properties"):
                try:
                    qprops = backend.qubit_properties
                    qprop = qprops(i) if callable(qprops) else qprops[i]
                except Exception:
                    qprop = None
            
            q_dict["T1"] = getattr(qprop, "t1", None) if qprop else None
            q_dict["T2"] = getattr(qprop, "t2", None) if qprop else None
            q_dict["frequency"] = getattr(qprop, "frequency", None) if qprop else None
            q_dict["anharmonicity"] = getattr(qprop, "anharmonicity", None) if qprop else None

            # 2. readout_error y readout_length desde la instrucción 'measure' del Target
            if target and "measure" in target and (i,) in target["measure"]:
                meas_prop = target["measure"][(i,)]
                q_dict["readout_error"] = getattr(meas_prop, "error", None)
                q_dict["readout_length"] = getattr(meas_prop, "duration", None)
            else:
                q_dict["readout_error"] = None
                q_dict["readout_length"] = None

            # 3. Propiedades adicionales que no todos los BackendV2 definen por estándar
            q_dict["prob_meas0_prep1"] = getattr(qprop, "prob_meas0_prep1", None) if qprop else None
            q_dict["prob_meas1_prep0"] = getattr(qprop, "prob_meas1_prep0", None) if qprop else None

            qubit_properties_list.append(q_dict)

    return qubit_properties_list

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
        loadAndRunExperiments()
    elif args.command == "show":
        readAndPlotExperiment(args.filepath)

if __name__ == '__main__':
    main()