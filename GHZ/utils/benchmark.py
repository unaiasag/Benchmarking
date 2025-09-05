import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli
from qiskit_ibm_runtime import Estimator, EstimatorOptions

def compute_l(epsilon, delta):
    '''
    l = 8 * log(4 / delta) / epsilon^2
    '''
    return int(np.ceil(8 * np.log(4 / delta) / epsilon**2))

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
        # 2. Sample a string from {I,Z}^{⊗n}, represented by binary bits
        z_bits = np.random.randint(0, 2, size=n)
        x_bits = np.zeros(n, dtype=int)
        pauli_base = Pauli((z_bits, x_bits))

        # 3. Conjugate by Clifford: G P G†
        pauli_conj = pauli_base.evolve(clifford, frame="s")

        # 4. Convert to string and phase
        pauli_str = pauli_conj.to_label()
        phase = pauli_conj.phase  # 0 = +1, 1 = i, 2 = -1, 3 = -i
        sign = [1, 1j, -1, -1j][phase]

        samples.append((pauli_str.strip("-"), sign))

    return samples

def run_circuit(circuit, observables, shots, mode):

    estimator = Estimator(mode=mode, options=EstimatorOptions(default_shots=shots))

    job = estimator.run([(circuit, obs) for obs in observables])

    result = job.result()
    
    return result
    
def run_GHZ_experiment(num_qubits, mode, transpiled_circuit, epsilon=0.01, delta=0.01):
    """
    Run a GHZ experiment to estimate the fidelity of a GHZ state.
    Args:
        num_qubits (int): Number of qubits in the GHZ state.
        mode (str): Execution mode for the estimator.
        transpiled_circuit (QuantumCircuit): Transpiled GHZ circuit.
        epsilon (float): Privacy parameter for differential privacy.
        delta (float): Privacy parameter for differential privacy.
    Returns:
        tuple: A tuple containing:
            - observables (list): List of selected stabilizer observables.
            - expected_vals (np.ndarray): Expected values from the circuit execution.
            - fidelity_estimate (float): Estimated fidelity of the GHZ state.
    """
    l = compute_l(epsilon=epsilon, delta=delta)  # Example values for epsilon and delta
    m_i = 1
    # Select random observables
    samples = sample_ghz_stabilizer(num_qubits, l)
    observables = [Pauli(p) for (p, _) in samples]
    signs = np.array([sign for (_, sign) in samples])
    observable_isa = [observable.apply_layout(layout=transpiled_circuit.layout) for observable in observables]

    result = run_circuit(transpiled_circuit, observable_isa, shots=m_i, mode=mode)
    expected_vals = np.array([pub_result.data.evs for pub_result in result])
    fidelity_estimate = np.mean(expected_vals * signs)
    
    return observables, expected_vals, fidelity_estimate