from braket.circuits import Circuit
import numpy as np

def create_lineal_ghz_circuit_2(num_qubits):
    ghz_circuit = Circuit()

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
                ghz_circuit.cnot(q, q - 1)
                new_entangled.append(q - 1)
            # Try to entangle q+1 (right)
            if q + 1 < num_qubits and (q + 1 not in entangled and q + 1 not in new_entangled):
                ghz_circuit.cnot(q, q + 1)
                new_entangled.append(q + 1)
        if not new_entangled:
            break  # No more qubits to entangle
        entangled.extend(new_entangled)

    return ghz_circuit

def create_exponential_ghz_circuit_2(num_qubits):
    ghz_circuit = Circuit()
    # Apply a Hadamard gate to the first qubit
    ghz_circuit.h(0)

    l = int(np.ceil(np.log2(num_qubits)))
    for i in range(l, 0, -1):
        for j in range(0, num_qubits, 2**i):
            if j + 2**(i-1) < num_qubits:
                ghz_circuit.cnot(j, j + 2**(i-1)) 
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

    elif mode == "lineal_v2":
        ghz_circuit = create_lineal_ghz_circuit_2(num_qubits)
    elif mode == "log_v2":
        ghz_circuit = create_exponential_ghz_circuit_2(num_qubits)
    else:
        raise ValueError("Invalid mode. Choose 'lineal_v2' or 'log_v2'.")
    

    return ghz_circuit