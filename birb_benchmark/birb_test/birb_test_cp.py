import random 
from typing import override

from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import random_clifford
from qiskit.converters import circuit_to_dag, dag_to_circuit

from birb_test import BiRBTest

class BiRBTestCP(BiRBTest):
    """
    Class for testing BiRB on circuits composed of layers where only a percentage
    of each layer consists of a Clifford circuit.

    Args:
        Same as BiRBTest, except for `percent`, which specifies the percentage
        of the Clifford layer used in each test circuit.
    """

    def __init__(self, qubits, depths, sim_type, backend_name, account_name ,circuits_per_depth=int(1e5), percent=0.5):
        super().__init__(qubits, depths,sim_type, backend_name, account_name, circuits_per_depth,)
        self.percent = percent


    @override
    def _generateRandomLayer(self):

        """
        Takes a Clifford operator, decomposes it into a specific circuit, and selects 
        a portion of it corresponding to `self.percent` of its total depth. 
        The starting point for extracting this subcircuit is chosen randomly.

        Returns:
            qc (QuantumCircuit): A circuit composed of `self.percent` percent of the 
            depth of a Clifford circuit on `self.n` qubits.
        """

        clifford_circuit = random_clifford(self.qubits).to_circuit()

        if(self.percent == 1.0):
            return clifford_circuit

        dag = circuit_to_dag(clifford_circuit)
        layers = list(dag.layers())
        total_layers = len(layers)
        half_depth = int(total_layers * self.percent)

        # Chooose starting point
        start_layer = random.randint(0, total_layers - half_depth)
        end_layer = start_layer + half_depth

        sub_dag = DAGCircuit()
        sub_dag.name = "subcircuit"
        sub_dag.add_qubits(dag.qubits)
        sub_dag.add_clbits(dag.clbits)

        for i in range(start_layer, end_layer):

            for node in layers[i]['graph'].op_nodes():
                sub_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

        sub_circuit = dag_to_circuit(sub_dag)
        return sub_circuit
