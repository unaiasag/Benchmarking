from qiskit import QuantumCircuit
import numpy as np

def create_lineal_ghz_circuit(num_qubits):
    # Apply CNOT gates between the first qubit and all other qubits
    ghz_circuit = QuantumCircuit(num_qubits)

    # Apply a Hadamard gate to the first qubit
    ghz_circuit.h(0)

    # Apply CNOT gates to create the GHZ state
    for i in range(num_qubits -1):
        ghz_circuit.cx(i, i+1)

    return ghz_circuit

def create_lineal_ghz_circuit_2(num_qubits):
    ghz_circuit = QuantumCircuit(num_qubits)

    # Determine middle qubit
    middle = num_qubits // 2

    # Apply Hadamard to the middle qubit
    ghz_circuit.h(middle)

    # List of "currently entangled" qubits
    entangled = [middle]

    # Iteratively apply CNOTs to expand entanglement symmetrically
    while True:
        new_entangled = []
        for q in entangled:
            # Try to entangle q-1 (left)
            if q - 1 >= 0 and (q - 1 not in entangled and q - 1 not in new_entangled):
                ghz_circuit.cx(q, q - 1)
                new_entangled.append(q - 1)
            # Try to entangle q+1 (right)
            if q + 1 < num_qubits and (q + 1 not in entangled and q + 1 not in new_entangled):
                ghz_circuit.cx(q, q + 1)
                new_entangled.append(q + 1)
        if not new_entangled:
            break  # No more qubits to entangle
        entangled.extend(new_entangled)

    return ghz_circuit

def create_exponential_ghz_circuit(num_qubits):
    ghz_circuit = QuantumCircuit(num_qubits)

    # Apply a Hadamard gate to the first qubit
    ghz_circuit.h(0)

    superposition_qubits = [0]
    non_superposition_qubits = list(range(1, num_qubits))

    while non_superposition_qubits:
        new_superposition_qubits = []
        for qubit in superposition_qubits:
            if not non_superposition_qubits:
                break
            qubit_to_superpose = non_superposition_qubits.pop(0)
            ghz_circuit.cx(qubit, qubit_to_superpose)
            new_superposition_qubits.append(qubit_to_superpose)
        superposition_qubits.extend(new_superposition_qubits)

    return ghz_circuit

def create_exponential_ghz_circuit_2(num_qubits):
    ghz_circuit = QuantumCircuit(num_qubits)
    # Apply a Hadamard gate to the first qubit
    ghz_circuit.h(0)

    l = int(np.ceil(np.log2(num_qubits)))
    for i in range(l, 0, -1):
        for j in range(0, num_qubits, 2**i):
            if j + 2**(i-1) < num_qubits:
                ghz_circuit.cx(j, j + 2**(i-1)) 
    return ghz_circuit


def create_ghz_circuit(num_qubits, mode="lineal"):
    """
    Create a GHZ (Greenberger-Horne-Zeilinger) state circuit for the specified number of qubits.

    Args:
        num_qubits (int): The number of qubits in the GHZ state.

    Returns:
        QuantumCircuit: A quantum circuit that prepares the GHZ state.
    """
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits.")

    if mode == "lineal":
        ghz_circuit = create_lineal_ghz_circuit(num_qubits)
    elif mode == "lineal_v2":
        ghz_circuit = create_lineal_ghz_circuit_2(num_qubits)
    elif mode == "log":
        ghz_circuit = create_exponential_ghz_circuit(num_qubits)
    elif mode == "log_v2":
        ghz_circuit = create_exponential_ghz_circuit_2(num_qubits)
    else:
        raise ValueError("Invalid mode. Choose 'lineal', 'lineal_v2', 'log', or 'log_v2'.")
    

    return ghz_circuit