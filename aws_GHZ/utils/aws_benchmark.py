import numpy as np
import logging

logger = logging.getLogger("qcentroid-user-log")

def required_shots_witnesses(N, eps, delta):
    return int((2/eps**2) * np.log(2*(N+1)/delta))

def estimate_P(counts, N):
    total = sum(counts.values())
    return (
        counts.get("0"*N, 0) +
        counts.get("1"*N, 0)
    ) / total

def parity(bitstring):
    return (-1)**(sum(int(b) for b in bitstring))

def estimate_Mk(counts):
    total = sum(counts.values())
    val = 0
    for b, c in counts.items():
        val += parity(b) * c
    return val / total

def estimate_C(all_counts, N):
    C = 0

    for k in range(1, N+1):
        Mk = estimate_Mk(all_counts[k])
        C += ((-1)**k) * Mk

    return C / N

def run_GHZ_Witnesses_experiment(circuits, num_qubits, shots, backend):
    logger.info(f"Running GHZ Witnesses experiment with {num_qubits} qubits and {shots} shots")
    
    all_counts = []
    metadata = []
    for circuit in circuits:
        result = backend.run(circuit, shots=shots).result()
        all_counts.append(result.measurement_counts)
        result_dict = {
            "measurement_counts": result.measurement_counts,
            "measurement_probabilities": result.measurement_probabilities,
            "task_id": result.task_metadata.id,
            "shots": result.task_metadata.shots,
            "device": result.task_metadata.deviceId        }
        metadata.append(result_dict)

    P = estimate_P(all_counts[0], num_qubits)
    C = estimate_C(all_counts, num_qubits)

    F = (P + C) / 2

    return F, P, C, metadata