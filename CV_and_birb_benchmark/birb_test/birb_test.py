import os
import pickle
from collections import Counter
import random
import sys
import statistics
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_pauli, Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator 
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2, Session
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2 

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from rich.align import Align

class BiRBTest(ABC):
    """
    Abstract class that encapsulates the BiRB benchmark for different types of
    circuits on an IBM processor, either real or simulated.
    """

    def __init__(self, qubits, depths, sim_type, execution_mode, backend_name, account_name, circuits_per_depth = int(1e5), shots_per_circuit = int(1e5)):
        """
        Constructor for the benchmark test class.

        Args:
            qubits (int): Number of qubits available on the target quantum processor.

            depths (list[int]): List of circuit depths to be tested.

            sim_type (str): Type of simulation to use. Options include:
                - "aer": Use Qiskit's AerSimulator with a noise model.
                - "fake": Use Qiskit's Fake Backends to simulate a real device.
                - "real": Use IBM's real devices.

            backend_name (str): Name of the IBM quantum backend (real or simulated) to run the tests on.

            account_name (str): Name of the IBM account to access the cloud.

            circuits_per_depth (int): Number of random circuits to generate and test for each depth.

            shots_per_circuit (int): Number of shots we make for each circuit.
        """

        self.qubits = qubits
        self.depths = depths
        self.circuits_per_depth = circuits_per_depth
        self.shots_per_circuit = shots_per_circuit
        self.backend_name = backend_name
        self.sim_type = sim_type
        self.execution_mode = execution_mode
        
        service = QiskitRuntimeService(name=account_name)

        if(self.sim_type == "aer"):
            try:
                backend = service.backend(backend_name)
            except Exception:
                print("Error: Backend " + backend_name + " not found.")
                sys.exit(1)

            noise_real_model = NoiseModel.from_backend(backend)

            # Create noisy simulator backend
            self.sim_noise = AerSimulator(noise_model=noise_real_model)
            self.passmanager = generate_preset_pass_manager(optimization_level=3, 
                                                            backend=self.sim_noise)

        elif(self.sim_type == "fake" or self.sim_type == "noiseless"):
            try:
                self.backend = FakeProviderForBackendV2().backend(backend_name)
            except Exception:
                print("Error: Backend " + backend_name + " not found.")
                sys.exit(1)

            self.backend.refresh(service)
            self.sampler = SamplerV2(self.backend)

        elif(self.sim_type == "real"):
            try:
                self.backend = service.backend(backend_name)
                self.sampler = SamplerV2(self.backend)
            except Exception:
                print("Error: Backend " + backend_name + " not found.")
                sys.exit(1)

        else:
            print("Select a valid simulator: 'fake', 'aer' or 'real'")
            sys.exit(1)



    def _getEigenVectorGates(self, gate, eigenvalue):
        """
        Given a single-qubit Pauli operator and one of its eigenvalues (+1 or -1), 
        returns a list of quantum gates that prepare an eigenstate corresponding to 
        the specified eigenvalue.

        Args:
            gate (str): A single-qubit Pauli gate, one of "X", "Y", "Z", or "I".
            
            eigenvalue (int): The desired eigenvalue of the state to prepare.
                              Must be either +1 or -1.

        Returns:
            str: A string of gate names that, when applied to |0⟩, prepare the
                 corresponding eigenstate.
        """

        X = ['H','XH']
        Y = ['HS','XHS']
        Z = ['I','X']

        if(gate == 'X'):
            if(eigenvalue == 1): return X[0]
            else: return X[1]

        if(gate == 'Y'):
            if(eigenvalue == 1): return Y[0]
            else: return Y[1]

        if(gate == 'Z'):
            if(eigenvalue == 1): return Z[0]
            else: return Z[1]
        
        return 'I'


    def _randomStabilizerState(self, pauli):
        """
        Generates a quantum circuit that prepares a random stabilizer state 
        corresponding to the given Pauli operator.

        Args:
            pauli (Pauli): The Pauli operator for which a stabilizer state should be generated.

        Returns:
            QuantumCircuit: A quantum circuit that prepares a stabilizer state of the specified Pauli operator.
        """

        qubits = pauli.num_qubits

        pauli_str = str(pauli)

        # We do not count identity operators or negative signs.
        effective_gates = 0
        for p in pauli_str:
            if(p != 'I' and p != '-'): effective_gates += 1 


        # Number of positive and negative eigenvalues to use, depending on the sign of
        # the Pauli operator.
        if(pauli_str[0] == '-'): 
            if(effective_gates == 1): num_negative = 1
            else: num_negative = random.choices(range(1, effective_gates, 2))[0]

        else: num_negative = random.choices(range(0, effective_gates, 2))[0]
        num_positive = effective_gates - num_negative
            
        qc = QuantumCircuit(qubits)

        # Remove the sign and reverse the order for implementation.
        if(pauli_str[0] in '+-'): pauli_str = pauli_str[0:]
        pauli_str = reversed(pauli_str)
            
        for i, p in enumerate(pauli_str):

            if(p == 'I'): continue

            # Choose whether the target eigenvalue is +1 or -1.
            selected_eigenvalue = random.choices([1,-1])[0]
            if(selected_eigenvalue == 1):
                if(num_positive > 0):
                    num_positive -= 1
                else:
                    selected_eigenvalue = -1
            else:
                if(num_negative > 0):
                    num_negative -= 1
                else:
                    selected_eigenvalue = 1


            # Select the circuit that prepares the eigenvector of p
            # corresponding to the selected_eigenvalue.
            s = self._getEigenVectorGates(p , selected_eigenvalue)
            for g in s:
                if(g == 'S'): qc.s(i)
                if(g == 'H'): qc.h(i)
                if(g == 'X'): qc.x(i)
        return qc


    def _prepareRandomPauli(self):
        """
        Generates a random Pauli operator along with a circuit that prepares
        one of its stabilizer states.

        Returns:
            pauli (Pauli): The randomly generated Pauli operator, including its
                           sign.

            stabilizer (QuantumCircuit): A circuit that prepares one of the
                                        stabilizer states of the Pauli operator.
        """


        # Generate a random Pauli operator different from the identity, with
        # either a positive or negative sign.
        pauli = random_pauli(self.qubits)
        while(str(pauli) == 'I'*self.qubits):
            pauli = random_pauli(self.qubits)

        if(random.choices([0,1])[0]):
            pauli = -pauli

        # Generate the circuit to prepare the Pauli operator.
        return pauli, self._randomStabilizerState(pauli)


    def _getEigenvalue(self, bitstring, pauli):
        """
        Given a bitstring and a Pauli operator, returns the eigenvalue.

        Args:
            bitstring (int or str): The bitstring represented as a decimal
            integer or binary string.

            pauli (Pauli): The Pauli operator for which to check the
            eigenvalue.

        Returns:
            int: The eigenvalue of the bitstring with respect to the Pauli
            operator, either +1 or -1.
        """

        if isinstance(bitstring, str): binary = bitstring
        else: binary = f'{bitstring:0{self.qubits}b}'

        str_pauli = str(pauli)
        final_value = 1

        if(str_pauli[0] == '-'): 
            str_pauli = str_pauli[1:]
            final_value = -1
        
        for j, p in enumerate(str_pauli):
            if(p != 'I' and binary[j] == '1'): final_value *= -1

        return final_value


    @abstractmethod
    def _generateRandomLayer(self):
        """
        Generates the circuit for each layer of the test. This is an abstract method
        that must be implemented by each subclass.
        """
        pass


    def _generateRandomCircuit(self, depth):
        """
        Generates a quantum circuit composed of 'depth' layers, each formed by
        single- and two-qubit Clifford gates.

        Returns:
            QuantumCircuit: The constructed quantum circuit.
        """
        qc = QuantumCircuit(self.qubits) 
        for _ in range(0, depth):
            clifford_circuit = self._generateRandomLayer()
            qc = qc.compose(clifford_circuit) 

        return qc

    def _pauliMeasurementCircuit(self, pauli):
        """
        Builds a quantum circuit to perform measurement in the specified Pauli basis.

        Args:
            pauli (Pauli): The Pauli operator defining the measurement basis.

        Returns:
            QuantumCircuit: A circuit that transforms states from the given Pauli basis 
            to the computational basis for measurement.
        """

        pauli_str = str(pauli)

        # Le quitamos el signo
        if(pauli_str[0] == '-'):
            pauli_str = pauli_str[1:]

        qc = QuantumCircuit(pauli.num_qubits)
        for i, p in enumerate(reversed(pauli_str)):
            if p == 'X':
                qc.h(i)
            elif p == 'Y':
                qc.sdg(i)
                qc.h(i)

        return qc


    def test(self):
        """
            Run several tests for debugging purposes.
        """

        # 1. Verify that the circuit preparing the stabilizer actually produces
        # a valid stabilizer state.
        print("1. Stabilizer test:")
        initial_pauli, estabilizer_circuit = self._prepareRandomPauli() 
        v = Statevector(estabilizer_circuit)
        s = v.evolve(initial_pauli)
        print("Is stabilizer ?: ", v == s)

        # 2. Verify that the Pauli matrix has been correctly evolved through the
        # randomly generated Clifford circuit. To do this, we take U|ψ⟩ from the circuit
        # and compute s'·U|ψ⟩, which should return the same vector since s' = U·s·U†.
        print("\n2. Random circuit evolution:")
        random_circuit = self._generateRandomCircuit(1)
        final_pauli = initial_pauli.evolve(random_circuit, frame='s')
        mid_circuit = estabilizer_circuit.compose(random_circuit)        
        v = Statevector(mid_circuit)
        s = v.evolve(final_pauli)
        print("Is the state generated by the circuit an stabilizer of s'?: ",
              v == s)

        # 3. Generate the final circuit, including the basis change for measurement.
        print("\n3. Measurement in the s' basis':")

        all_right = True

        # Generate the final circuit
        final_circuit = estabilizer_circuit.compose(random_circuit)\
                                           .compose(self._pauliMeasurementCircuit(final_pauli))

        # Obtain the final state vector and the probability of each bitstring.
        v = Statevector(final_circuit).probabilities()

        # Verify that the bitstrings with probability 1 correspond to those
        # with eigenvalue +1.
        EPS = 1e-20
        for i, prob in enumerate(v):
            if(prob > EPS):
                if(self._getEigenvalue(i, final_pauli) == -1):
                    print("Error: Obtained -1")
                    print("Index: ", i)
                    print("Probability: ", prob)
                    print(v)
                    all_right = False

        print("All right?", all_right)
                


    def _runCircuitSimulation(self, circuit):
        """
        Given a quantum circuit, transpiles and executes it on the specified backend.
            
        Args:
            circuit (QuantumCircuit): circuit to transpile and execute in the backend.
            
        Returns:
            count_sim (dict): A dictionary where each key is a bitstring and each value is 
            the number of times that bitstring was observed.
        """

        transpiled_circuit = transpile(circuit, self.backend, optimization_level=3)
        if(self.sim_type == 'fake'):
            pub_result = self.sampler.run([transpiled_circuit],
                                          shots=self.shots_per_circuit).result()[0]

            counts_sim = pub_result.data.meas.get_counts()

        elif(self.sim_type == 'noiseless'):

            simulator = AerSimulator(
                basis_gates=self.backend.configuration().basis_gates,
                coupling_map=self.backend.configuration().coupling_map
            )

            result = simulator.run(transpiled_circuit, 
                                   shots=self.shots_per_circuit).result()
            counts_sim = result.get_counts(0)
            
        else:
            circuit_noise = self.passmanager.run(circuit) 
            result_sim = self.sim_noise.run(circuit_noise,
                                            shots=self.shots_per_circuit).result()
            counts_sim = result_sim.get_counts(0)

        return counts_sim


    def _runCircuit(self, depth):
        """
        Creates and executes a complete quantum circuit of the specified depth,
        performing multiple shots on it.

        Args:
            depth (int): Number of layers of the circuit.

        Returns:
            float: The average of the output eigenvalues obtained from the circuit.
        """

        # Initial Pauli and stabilizer state
        initial_pauli, estabilizer_circuit = self._prepareRandomPauli() 

        # Random circuit
        random_circuit = self._generateRandomCircuit(depth)

        # Pauli for measurement
        final_pauli = initial_pauli.evolve(random_circuit, frame='s') 

        # Complete circuit
        final_circuit = (
            estabilizer_circuit
            .compose(random_circuit)
            .compose(self._pauliMeasurementCircuit(final_pauli)))

        final_circuit.measure_all()
 
        # Run the circuit 
        counts_sim = self._runCircuitSimulation(final_circuit)

        mean = 0 
        total_num_shots = 0
        for bitstring, count in counts_sim.items():
            mean += self._getEigenvalue(bitstring, final_pauli) * count
            total_num_shots += count

        # Check that the computer is making the correct number of shots
        assert self.shots_per_circuit == total_num_shots, "Number of shots less than expected"
        mean /= self.shots_per_circuit

        return mean

    def _generateCircuit(self, depth):
        """
        Creates a quantum circuit with the specified depth.

        Args:
            depth (int): Number of layers of the circuit.

        Returns:
            final_circuit (QuantumCircuit): Circuit including preparation, random layers,
                                            and measurement.

            final_pauli (Pauli): The Pauli operator used for final measurement.
        """

        # Initial Pauli and stabilizer state
        initial_pauli, estabilizer_circuit = self._prepareRandomPauli() 

        # Random circuit
        random_circuit = self._generateRandomCircuit(depth)

        # Pauli for measurement
        final_pauli = initial_pauli.evolve(random_circuit, frame='s') 

        # Complete circuit
        final_circuit = (
            estabilizer_circuit
            .compose(random_circuit)
            .compose(self._pauliMeasurementCircuit(final_pauli)))

        final_circuit.measure_all()

        return final_circuit, final_pauli
    
    def _selectTranspileLayout(self):
        """
        Selects the most repeated layout in the transpilation of various 
        circuits (len(depths) x 100).

        Returns:
            list[int]: A layout (mapping of logical to physical qubits) that appears
                       most frequently after transpiling test circuits.
        """

        # Generate and transpile the circuits 
        circuits = []
        for depth in self.depths:
            for _ in range(100):
                circuit, _ = self._generateCircuit(depth)
                circuits.append(circuit)
        
        transpiled_circuits = transpile(circuits, self.backend, optimization_level=3)

        # Count layout repetitions
        counter = Counter()
        for circuit in transpiled_circuits:
            layout = tuple(circuit.layout.final_index_layout())
            layout_set = frozenset(layout)
            counter[layout_set] += 1

        # Return the most common layout
        most_repeated_set, _ = counter.most_common(1)[0]
        
        return list(most_repeated_set)  


    
    def _processBatchResults(self, results, paulis):
        """
        Computes the average of the eigenvalues for all the PUBs in a job.

        Args:
            results (Result): Result object obtained from a job with the 
                              result() method.
            
            paulis (list[Pauli]): List of pauli operators for the eigenvalue
                                  calculation.


        Returns:
            evs (list[float]): An average eigenvalue per circuit, computed 
                               from counts and Pauli measurement.
        """

        evs = []
        for i in range(len(results)):
            counts_sim = results[i].data.meas.get_counts()

            mean = 0 
            total_num_shots = 0
            for bitstring, count in counts_sim.items():
                mean += self._getEigenvalue(bitstring, paulis[i]) * count
                total_num_shots += count

            # Check that the computer is making the correct number of shots
            assert self.shots_per_circuit == total_num_shots, "Number of shots less than expected"
            mean /= self.shots_per_circuit
            evs.append(mean)

        return evs

    def prepareCircuits(self, file_prefix):

        """
        Creates, transpiles and saves all the circuits and paulis for all depths.

        Args:
            file_prefix (str): File name prefix (e.g. '.../50percent'), to which 
                               '_depth_X.pk' will be appended.
        """

        layout = self._selectTranspileLayout()
        
        console = Console()
        console.print("")

        panel = Panel(
            Align.center("[bold]🚀 Preparing Clifford circuits for "
                         "different depths[/bold]"),
            title="PROCESSING",
            border_style="green",
        )
        console.print(Align.center(panel))

        with Progress(
            TextColumn("[bold green]{task.fields[title]}"),
            BarColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("{task.fields[result]}"),
            TimeElapsedColumn(),
            transient=False
        ) as progress:
            overall_task = progress.add_task("", 
                                             total=len(self.depths),
                                             title="Total depths", 
                                             result="")
            
            for depth in self.depths:

                circuits, paulis = [], []
                for _ in range(self.circuits_per_depth):

                    final_circuit, final_pauli = self._generateCircuit(depth)
                    circuits.append(final_circuit)
                    paulis.append(final_pauli)

                transpiled_circuits = transpile(circuits, self.backend, optimization_level=3, initial_layout=layout)

                with open(file_prefix + f"_depth_{depth}.pk", "wb") as f:
                    pickle.dump((transpiled_circuits, paulis), f)

                progress.update(overall_task, advance=1)


    def run(self, eps=1e-4, file_prefix=None):

        """
        Runs the test using the provided data. If the results fall below the specified
        threshold, further execution is stopped.

        Args:
            eps (float): Tolerance threshold. If the result of an execution is
                         less than this value, no additional depths are tested.
            
            file_prefix (str): Folder containing the transpiled circuits 
                               for 'real' execution.

        Returns:
            results_per_depth (list[list[float]]): A list where each element
                                                   contains the results for a
                                                   specific depth.

            valid_depths (list[int]): A list of depths for which the test was
                                     actually executed.
        """

        results_per_depth = []
        valid_depths = []

        console = Console()
        console.print("")

        panel = Panel(
            Align.center("[bold]🚀 Running Clifford circuits for "
                         "different depths[/bold]"),
            title="PROCESSING",
            border_style="green",
        )
        console.print(Align.center(panel))

        with Progress(
            TextColumn("[bold green]{task.fields[title]}"),
            BarColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("{task.fields[result]}"),
            TimeElapsedColumn(),
            transient=False
        ) as progress:
            overall_task = progress.add_task("", 
                                             total=len(self.depths),
                                             title="Total depths", 
                                             result="")
            
            if (self.sim_type == "real"):

                if (self.execution_mode == "session"):
                    session = Session(self.backend)
                    self.sampler = SamplerV2(mode=session)

                for depth in self.depths:
                    
                    filepath = file_prefix + f"_depth_{depth}.pk"
                    try:
                        with open(filepath, "rb") as f:
                            circuits, paulis = pickle.load(f)
                    except Exception as e:
                        if (self.execution_mode == "session"): session.close()
                        print(f"Error loading file {filepath}: {e}")
                        print("Session closed")
                        sys.exit(1)
                    
                    results = self.sampler.run(circuits, shots=self.shots_per_circuit).result()
                    depth_result = self._processBatchResults(results, paulis)
                    results_per_depth.append(depth_result)
                    valid_depths.append(depth)
                    progress.update(overall_task, advance=1)

                    # If it is so low depth we not continue
                    if(statistics.mean(depth_result) < eps):
                        if (self.execution_mode == "session"): session.close()
                        break

                if (self.execution_mode == "session"): session.close()
                                      
            else:
                for depth in self.depths:
                    circuit_task = progress.add_task(f"{depth}",
                                                    total=self.circuits_per_depth,
                                                    title=f"Circuits of depth {depth}",
                                                    result="")

                    depth_result = []
                    for _ in range(self.circuits_per_depth):
                        result = self._runCircuit(depth)
                        depth_result.append(result)
                        progress.update(circuit_task,
                                        advance=1,
                                        result=f"[dim]Result:[/dim] {result:.4f}")

                    results_per_depth.append(depth_result)
                    valid_depths.append(depth)

                    progress.update(overall_task, advance=1)
                    progress.remove_task(circuit_task)

                    # If it is so low depth we not continue
                    if(statistics.mean(depth_result) < eps):
                        break


        return results_per_depth, valid_depths