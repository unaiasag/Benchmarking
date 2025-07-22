import numpy as np
import random
import itertools

from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, Estimator, EstimatorOptions

from .circuits import create_ghz_circuit

def compute_l(epsilon, delta):
    '''
    l = 8 * log(4 / delta) / epsilon^2
    '''
    return int(np.ceil(8 * np.log(4 / delta) / epsilon**2))

def pauli_product(p1, p2):
    """
    Multiply two Pauli operators (I, X, Y, Z) and return:
    - the result (I, X, Y, Z)
    - the phase (±1, ±i), but we ignore imaginary phase for DFE
    """
    # Pauli multiplication table: result and sign
    table = {
        ('I', 'I'): ('I', 1), ('I', 'X'): ('X', 1), ('I', 'Y'): ('Y', 1), ('I', 'Z'): ('Z', 1),
        ('X', 'I'): ('X', 1), ('X', 'X'): ('I', 1), ('X', 'Y'): ('Z', 1j), ('X', 'Z'): ('Y', -1j),
        ('Y', 'I'): ('Y', 1), ('Y', 'X'): ('Z', -1j), ('Y', 'Y'): ('I', 1), ('Y', 'Z'): ('X', 1j),
        ('Z', 'I'): ('Z', 1), ('Z', 'X'): ('Y', 1j), ('Z', 'Y'): ('X', -1j), ('Z', 'Z'): ('I', 1),
    }
    result, phase = table[(p1, p2)]
    return result, phase

def multiply_pauli_strings(s1, s2):
    """
    Multiply two full Pauli strings and return:
    - the resulting string
    - the overall sign (±1) ignoring imaginary components
    """
    result = []
    phase = 1
    for p1, p2 in zip(s1, s2):
        r, ph = pauli_product(p1, p2)
        result.append(r)
        phase *= ph
    sign = 1 if phase.real > 0 else -1  # Only track ±1
    return ''.join(result), sign

def ghz_stabilizer_generators(n):
    """
    Return the GHZ_n stabilizer generators.
    - One global X generator: X⊗X⊗...⊗X
    - n-1 Z pair generators: Z_i Z_{i+1}
    """
    generators = []

    # X⊗X⊗...⊗X
    generators.append('X' * n)

    # Z_i Z_{i+1}
    for i in range(n - 1):
        g = ['I'] * n
        g[i] = 'Z'
        g[i + 1] = 'Z'
        generators.append(''.join(g))

    return generators

def generate_ghz_stabilizer_group(n):
    """
    Return a dictionary {pauli_str: sign} for the GHZ stabilizer group.
    """
    generators = ghz_stabilizer_generators(n)
    num_gens = len(generators)

    group = {}
    for binary in itertools.product([0, 1], repeat=num_gens):
        current = 'I' * n
        sign = 1
        for i, b in enumerate(binary):
            if b:
                current, s = multiply_pauli_strings(current, generators[i])
                sign *= s
        group[current] = sign

    return group

def sample_ghz_stabilizer_group_with_signs(n, num_samples):
    """
    Generate `num_samples` random stabilizers (Pauli strings + signs)
    from the GHZ_n stabilizer group.
    Returns a dict {pauli_str: sign}.
    """
    generators = ghz_stabilizer_generators(n)
    num_gens = len(generators)
    
    stabilizer_samples = {}
    max_tries = num_samples * 10

    while len(stabilizer_samples) < num_samples and max_tries > 0:
        max_tries -= 1
        # Random binary vector selecting generators
        subset = [random.choice([0,1]) for _ in range(num_gens)]

        current = 'I' * n
        sign = 1
        for i, bit in enumerate(subset):
            if bit:
                current, s = multiply_pauli_strings(current, generators[i])
                sign *= s

        stabilizer_samples[current] = sign

    return stabilizer_samples

def select_stabilizer_observables(n, num_observables):
    stabilizers = sample_ghz_stabilizer_group_with_signs(n, num_observables)
    observables = list(stabilizers.keys())
    if len(stabilizers) < num_observables:
        selected_obs = random.choices(observables, k=num_observables)
        selected_signs = [stabilizers[obs] for obs in selected_obs]
        return [SparsePauliOp.from_list([(p, 1.0)]) for p in selected_obs], np.array(selected_signs)
    else:
        return [SparsePauliOp.from_list([(p, 1.0)]) for p in observables], np.array(list(stabilizers.values()))

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
    observables, signs = select_stabilizer_observables(num_qubits, l) 
    observable_isa = [observable.apply_layout(layout=transpiled_circuit.layout) for observable in observables]

    result = run_circuit(transpiled_circuit, observable_isa, shots=m_i, mode=mode)
    expected_vals = np.array([pub_result.data.evs for pub_result in result])
    fidelity_estimate = np.mean(expected_vals * signs)
    
    return observable_isa, expected_vals, fidelity_estimate