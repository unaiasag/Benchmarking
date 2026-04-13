import numpy as np

from datetime import datetime

from utils.aws_benchmark import required_shots_witnesses, run_GHZ_Witnesses_experiment
from utils.aws_circuits import create_ghz_circuit

from braket.aws import AwsDevice

import logging

logger = logging.getLogger("qcentroid-user-log")

class GHZExperiment():
    def __init__(self, num_qubits, circuit, l, m_i, backend):
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

    def prepare_experiment(self):

        N = self.num_qubits
        circuits = [self.circuit.copy()]

        # COHERENCE
        for k in range(1, N + 1):
            circuit_ = self.circuit.copy()

            theta_k = k * np.pi / N

            for qubit in range(N):
                circuit_.rz(qubit, -theta_k)     # Rz(-θk)
                circuit_.ry(qubit, -np.pi / 2)  # Ry(-π/2)

            circuits.append(circuit_)

        self.pubs = circuits
        
    def run_experiment(self):
        return run_GHZ_Witnesses_experiment(circuits=self.pubs, num_qubits=self.num_qubits, shots=self.m_i, backend=self.backend)

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

def runExperiments(backend_arn, backend_architecture, numero_qubits_inicial, numero_maximo_qubits, epsilon, delta):
    # Log the start of the experiment (backend, qubits, epsilon, delta)
    logger.info(f"Backend: {backend_arn}, Qubits: {numero_qubits_inicial}, Epsilon: {epsilon}, Delta: {delta}")

    backend = AwsDevice(backend_arn)
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
        """circuit_name = f"GHZ_{num_qubits}q"

        # on test save the circuit image as well
        circuit_file_image = os.path.join(circuits_folder, f"{circuit_name}.png")
        diagram = circuit.diagram()

        plt.figure(figsize=(6,2))
        plt.text(0.01, 0.5, diagram, family="monospace")
        plt.axis("off")
        plt.savefig(circuit_file_image, dpi=300)
        plt.close()"""
        
        l = None # l is not needed for the Witnesses benchmark, but we keep it as a parameter for simplicity
        m_i = required_shots_witnesses(num_qubits, epsilon, delta)
        experiment = GHZExperiment(num_qubits, circuit, l, m_i, backend)
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
                fidelity_estimate, P, C, metadata = result

                results_to_save.append(
                    (num_qubits, P, C, fidelity_estimate, metadata)
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
            num_qubits, P, C, fidelity_estimate, metadata = results
            result_dict[f"{num_qubits}_qubits"] = {
                "P": P,
                "C": C,
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

    return runExperiments(backend_arn, backend_architecture, numero_qubits_inicial, numero_maximo_qubits, epsilon, delta)
