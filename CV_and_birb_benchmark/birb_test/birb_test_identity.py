import random
from typing import override

from qiskit import QuantumCircuit

from birb_test import BiRBTest

class BiRBTestId(BiRBTest):
    """
    Class for testing BiRB on circuits composed of layers containing the identity.

    Args:
        Same as BiRBTest 
    """

    def __init__(self, qubits, depths, sim_type, backend_name, account_name, circuits_per_depth=int(1e5), shots_per_circuit=int(1e5)):
        super().__init__(qubits, depths,sim_type, backend_name, account_name, circuits_per_depth, shots_per_circuit)

    @override
    def _generateRandomLayer(self):
        """
        Returns a layer formed of identity

        Returns:
            qc (QuantumCircuit): The identity circuit
            """
        return QuantumCircuit(self.qubits)
