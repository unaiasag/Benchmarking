import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli



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

def run_circuit(backend, pubs, observables, shots, execution_mode="vqc"):
    """
    Run the given circuit with the specified observables using IBM Quantum Runtime.
    Args:
        backend: The quantum backend to execute the circuit on.
        pubs (list): List of ParameterizedQuantumCircuits to execute.
        observables (list): List of Pauli observables to measure.
        shots (int): Number of shots for the execution.
        execution_mode (str): "estimator" or "sampler" or "vqc".
        tags (list): List of tags to attach to the job.
    Returns:
        Result object from the execution.
    """
    evs = None

    if execution_mode == "vqc":
        circuits = []
        if isinstance(pubs, list) or isinstance(pubs, tuple):
            pub = pubs[0]
            if isinstance(pub, tuple):
                circuit_base = pub[0]
                if len(pub) > 1 and pub[1] is not None:
                    for parameter_set in pub[1]:
                        circuits.append(circuit_base.copy().assign_parameters(parameter_set))
            else:
                circuit = pub
        else:
            circuit = pubs

        # 4. Ejecutar el circuito en la QPU de QMIO
        bitstrings = []
        for circuit in circuits:
            job = backend.run(circuit, shots=shots)
            result = job.result()
            counts = result.get_counts() # Devuelve un dict: {'00': 450, '11': 550}
            bitstrings.append([b for b in counts.keys()][0])
        evs = []
        for obs_index, observable in enumerate(observables):
            bitstring = bitstrings[obs_index]
            expval = 0
            parity = 1
            for qubit, pauli in enumerate(observable.to_label()):
                if pauli != 'I' and bitstring[qubit] == '1':
                    parity *= -1
 
            expval += parity
            evs.append(expval)
            
        return np.array(evs)
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}. Supported modes are 'estimator', 'sampler', and 'vqc'.")
    
def run_GHZ_experiment(backend, pubs, num_qubits, l, observable_isa, signs, m_i, execution_mode="vqc"):
    """
    Run a GHZ experiment to estimate the fidelity of a GHZ state.
    Args:
        backend: The quantum backend to execute the circuit on.
        num_qubits (int): Number of qubits in the GHZ state.
        transpiled_circuit (QuantumCircuit): Transpiled GHZ circuit.
        epsilon (float): Privacy parameter for differential privacy.
        delta (float): Privacy parameter for differential privacy.
    Returns:
        tuple: A tuple containing:
            - observables (list): List of selected stabilizer observables.
            - expected_vals (np.ndarray): Expected values from the circuit execution.
            - fidelity_estimate (float): Estimated fidelity of the GHZ state.
    """

    max_pub_size = 9_500_000
    observable_size = 1#4 ** num_qubits
    num_partitions =  np.ceil(l * observable_size / max_pub_size).astype(int)

    all_expected_vals = np.zeros(l)

    tags = [f"{num_qubits}qb", f"l={l}"]

    for i in range(num_partitions):
        tags_i = tags + [f"part={i+1}/{num_partitions}"]
        start_idx = i * (l // num_partitions)
        end_idx = min((i + 1) * (l // num_partitions), l)
        partition_observables = observable_isa[start_idx:end_idx]
        expected_vals = run_circuit(backend=backend, pubs=pubs, observables=partition_observables, shots=m_i, execution_mode=execution_mode)
        all_expected_vals[start_idx:end_idx] = expected_vals

    fidelity_estimate = np.mean(all_expected_vals * signs)
    
    return all_expected_vals, fidelity_estimate