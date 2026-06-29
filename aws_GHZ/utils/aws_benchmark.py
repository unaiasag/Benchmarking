import numpy as np
import logging

from braket.circuits import Observable
from braket.program_sets import ProgramSet, CircuitBinding


from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli

logger = logging.getLogger("qcentroid-user-log")

def compute_l(epsilon, delta):
    '''
    l = 8 * log(4 / delta) / epsilon^2
    '''
    return int(np.ceil(8 * np.log(4 / delta) / epsilon**2))

def observable_from_string(s: str) -> Observable:
    gates = {
        "I": Observable.I,
        "X": Observable.X,
        "Y": Observable.Y,
        "Z": Observable.Z
    }
    return Observable.TensorProduct([gates[pauli](idx) for idx, pauli in enumerate(s)])

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

def ghz_circuit(n):
    """
    Create a Clifford circuit that generates GHZ state from |0...0⟩.
    """
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    return qc

def sample_ghz_stabilizer(n, num_samples=1):
    """
    Sample `num_samples` Pauli strings from the GHZ_n stabilizer group.
    
    Returns a list of (Pauli, sign) tuples.
    """
    # 1. Build the GHZ Clifford circuit
    qc = ghz_circuit(n)
    clifford = Clifford(qc)

    samples = []
    for _ in range(num_samples):
        not_identity = False
        while not_identity == False:
            # 2. Sample a string from {I,Z}^{⊗n}, represented by binary bits
            z_bits = np.random.randint(0, 2, size=n)
            x_bits = np.zeros(n, dtype=int)
            pauli_base = Pauli((z_bits, x_bits))

            # 3. Conjugate by Clifford: G P G†
            pauli_conj = pauli_base.evolve(clifford, frame="s")

            if pauli_conj.to_label() != 'I' * n:
                not_identity = True

        # 4. Convert to string and phase
        pauli_str = pauli_conj.to_label()
        phase = pauli_conj.phase  # 0 = +1, 1 = i, 2 = -1, 3 = -i
        sign = [1, 1j, -1, -1j][phase]

        samples.append((pauli_str.strip("-"), sign))

    return samples


def run_GHZ_RF_experiment(circuit, num_qubits, observables, signs, backend, min_shots=1):
    logger.info(f"Running GHZ RF experiment with {num_qubits} qubits and {len(observables)} observables")
    
    program_set = ProgramSet(
        CircuitBinding(
            circuit=circuit,
            observables=observables,
        ),
        shots_per_executable = len(observables)
    )

    subs, index_map = program_set.split(1_000_000)
    all_expected_vals = np.zeros(len(observables))
    task_ids = []
    total_shots = 0
    device_id = None
    for sub, indices in zip(subs, index_map):
        result_set = backend.run(sub, shots=sub.total_executables*min_shots).result()

        expected_vals = [result.expectation for result in result_set[0]]
        all_expected_vals[indices] = expected_vals

        task_ids.append(result_set.task_metadata.id)
        total_shots += result_set.task_metadata.successfulShots
        if device_id is None:
            device_id = result_set.task_metadata.deviceId

    fidelity_estimate = np.mean(all_expected_vals * signs)
    
    result_dict = {
        "expected_values": all_expected_vals,
        "task_ids": task_ids,
        "shots": total_shots,
        "device": device_id
    }

    return fidelity_estimate, result_dict