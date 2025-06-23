import random
from typing import override

from qiskit.quantum_info import random_clifford
from qiskit import QuantumCircuit

from birb_test_aws import BiRBTest

class BiRBTest2q1q(BiRBTest):
    """
    Class for testing BiRB on circuits composed of layers containing only random
    single- and two-qubit Clifford gates.

    Args:
        Same as BiRBTest, except for `prob2q`, which specifies the density of 
        two-qubit gates in each layer.
    """

    def __init__(self, qubits, depths, sim_type, backend_name, account_name, circuits_per_depth=int(1e5), shots_per_circuit=int(1e5), prob2q=0.5):
        super().__init__(qubits, depths, sim_type, backend_name, account_name, circuits_per_depth, shots_per_circuit)
        self.prob2q = prob2q

    @override
    def _generateRandomLayer(self):
        """
        Returns a layer formed of sigle and two qubit gates

        Returns:
            qc (QuantumCircuit): A circuit composed of single and two-qubit Clifford gates,
            applied in parallel within each layer.
        """

        available_qubits = [x for x in range(0, self.qubits)]

        qc = QuantumCircuit(self.qubits)

        while(len(available_qubits) > 0):

            # Choose randomly one ore two qubit gates
            if(len(available_qubits) == 1): result = '1Q'
            else: result = random.choices(['1Q', '2Q'], weights=[1 - self.prob2q, self.prob2q])[0]

            if(result == '1Q'):

                # Select one qubit gate randomly
                qubit = available_qubits.pop(random.randrange(len(available_qubits)))
                gate = random_clifford(1).to_instruction()

                qc.append(gate , [qubit])
            else:

                # Select two qubits gate randomly
                qubit1 = available_qubits.pop(random.randrange(len(available_qubits)))
                qubit2 = available_qubits.pop(random.randrange(len(available_qubits)))
                gate = random_clifford(2).to_instruction()

                qc.append(gate , [qubit1, qubit2])

        return qc
