import numpy as np

from datetime import datetime

from utils.aws_benchmark import required_shots_witnesses, run_GHZ_Witnesses_experiment, sample_ghz_stabilizer, observable_from_string, compute_l, run_GHZ_RF_experiment
from utils.aws_circuits import create_ghz_circuit

from braket.aws import AwsDevice
from QCentroidLoaders import BraketLoader

import logging

logger = logging.getLogger("qcentroid-user-log")

class GHZExperiment():
    def __init__(self, num_qubits, circuit, l, m_i, backend, min_shots=1):
        """
        Docstring for __init__
        :param num_qubits: Number of qubits in the GHZ state
        :param circuit: QuantumCircuit that prepares the GHZ state
        :param l: Parameter for the experiment
        :param m_i: Number of shots for the experiment
        :param backend: Backend for running the experiment
        """ 
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.l = l
        self.m_i = m_i
        self.backend = backend
        self.min_shots = min_shots
    def prepare_experiment(self):

        samples = sample_ghz_stabilizer(self.num_qubits, self.l)
        observables = [observable_from_string(p) for (p, _) in samples]
        signs = np.array([sign for (_, sign) in samples])

        self.observables = observables
        self.signs = signs
        
    def run_experiment(self):
        return run_GHZ_RF_experiment(circuit=self.circuit, num_qubits=self.num_qubits, observables=self.observables, signs=self.signs, backend=self.backend, min_shots=self.min_shots)

def getBestGHZCircuitsPerQPU(backend_architecture, start_qubits=2, maximo=2):
    """
    Transpila y selecciona los mejores circuitos según la calibración del backend.

    Args:
        backend_architecture (str): Arquitectura del backend, puede ser "lineal" o "fully-connected"
        qubits (int): Número de qubits
        calibrations (dict): Calibraciones del backend
        limit (int): Número máximo de circuitos a seleccionar

    Returns:
        List[QuantumCircuit]: Circuitos transpileados optimizados
    """

    if backend_architecture == "lineal":
        best_strategy = "lineal_v2"
    elif backend_architecture == "fully-connected":
        best_strategy = "log_v2"

    circuits = {}

    for i in range(start_qubits, maximo+1):
        circuits[i] = create_ghz_circuit(i, mode=best_strategy)

    return circuits

def runExperiments(backend_arn, backend_architecture, numero_qubits_inicial, numero_maximo_qubits, epsilon, delta, min_shots=1):
    # Log the start of the experiment (backend, qubits, epsilon, delta)
    logger.info(f"Backend: {backend_arn}, Qubits: {numero_qubits_inicial}, Epsilon: {epsilon}, Delta: {delta}")

    backend = BraketLoader.get_target()
    backend_qubits = numero_maximo_qubits

    start_date = datetime.now()
    start_date_str = start_date.strftime("%Y%m%d_%H%M%S")

    maximo = numero_maximo_qubits

    best_circuits = getBestGHZCircuitsPerQPU(backend_architecture=backend_architecture, start_qubits=numero_qubits_inicial, maximo=maximo)

    experiments = {}
    smaller_size = numero_qubits_inicial
    for num_qubits in range(smaller_size, maximo + 1):
        # Save the circuit
        circuit = best_circuits[num_qubits]
        
        l = compute_l(epsilon, delta)
        m_i = None # m_i is not needed for the RF benchmark, but we keep it as a parameter for simplicity
        experiment = GHZExperiment(num_qubits, circuit, l, m_i, backend, min_shots=min_shots)
        experiment.prepare_experiment()
        experiments[num_qubits] = experiment
    
    num_qubits = numero_qubits_inicial - 1
    results_to_save = []

    try:
        max_idx = maximo
        min_idx = smaller_size
        idx = int((max_idx+min_idx) // 2)
        end = False

        while not end:
            experiment = experiments[idx]
            try:
                # Run the GHZ experiment
                result = experiment.run_experiment()

                num_qubits = experiment.num_qubits
                fidelity_estimate, metadata = result

                results_to_save.append(
                    (num_qubits, fidelity_estimate, metadata)
                )

                if fidelity_estimate < (1 / 2 - epsilon):
                    # If the fidelity is below the threshold discard the upper half of the search space
                    max_idx = idx - 1
                    if min_idx > max_idx:
                        logger.info(
                            f"Fidelity estimate {fidelity_estimate} crosses threshold at {idx} qubits. Stopping further experiments."
                        )
                        end = True
                    else:
                        idx = int((max_idx+min_idx) // 2)
                else:
                    # If the fidelity is above the threshold discard the lower half of the search space
                    min_idx = idx + 1
                    if min_idx > max_idx:
                        logger.info(
                            f"Fidelity estimate {fidelity_estimate} crosses threshold at {idx+1} qubits. Stopping further experiments."
                        )
                        end = True
                    else:
                        idx = int((max_idx+min_idx) // 2)


            except Exception as exp_err:
                logger.error(f"Error during experiment execution: {exp_err}")
                end = True  # stop further experiments but still save results

    except Exception as session_err:
        logger.error(f"Session-level error occurred: {session_err}")

    finally:
        # ALWAYS save whatever results we have
        result_dict = {
            "backend": backend_arn,
        }
        for results in results_to_save:
            num_qubits, fidelity_estimate, metadata = results
            result_dict[f"{num_qubits}_qubits"] = {
                "fidelity_estimate": fidelity_estimate,
                "epsilon": epsilon,
                "delta": delta,
                "metadata": metadata,
            }
    return result_dict
    

def run(input_data:dict, solver_params:dict, extra_arguments:dict) -> dict:
    backend_arn = input_data.get("backend_arn")
    backend_architecture = input_data.get("backend_architecture")
    numero_qubits_inicial = input_data.get("numero_qubits_inicial")
    numero_maximo_qubits = input_data.get("numero_maximo_qubits")
    epsilon = input_data.get("epsilon")
    delta = input_data.get("delta")
    min_shots = input_data.get("min_shots", 1)

    return runExperiments(backend_arn, backend_architecture, numero_qubits_inicial, numero_maximo_qubits, epsilon, delta, min_shots=min_shots)
