from qiskit import QuantumCircuit
from qiskit.circuit.library import standard_gates
from qiskit.quantum_info import StabilizerState, Operator # https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.StabilizerState
# For the stabilizer this object may be useful https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Clifford
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp, Clifford
#from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _BASIS_1Q, _BASIS_2Q
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

gates_1q_data = [
    (standard_gates.IGate, 1, 0),
    (standard_gates.SXGate, 1, 0),
    (standard_gates.XGate, 1, 0),
    (standard_gates.RZGate, 1, 1),
    (standard_gates.RGate, 1, 2),
    (standard_gates.HGate, 1, 0),
    (standard_gates.PhaseGate, 1, 1),
    (standard_gates.RXGate, 1, 1),
    (standard_gates.RYGate, 1, 1),
    (standard_gates.SGate, 1, 0),
    (standard_gates.SdgGate, 1, 0),
    (standard_gates.SXdgGate, 1, 0),
    (standard_gates.TGate, 1, 0),
    (standard_gates.TdgGate, 1, 0),
    (standard_gates.UGate, 1, 3),
    (standard_gates.U1Gate, 1, 1),
    (standard_gates.U2Gate, 1, 2),
    (standard_gates.U3Gate, 1, 3),
    (standard_gates.YGate, 1, 0),
    (standard_gates.ZGate, 1, 0),
]

gates_2q_data = [
    (standard_gates.CXGate, 2, 0),
    (standard_gates.DCXGate, 2, 0),
    (standard_gates.CHGate, 2, 0),
    (standard_gates.CPhaseGate, 2, 1),
    (standard_gates.CRXGate, 2, 1),
    (standard_gates.CRYGate, 2, 1),
    (standard_gates.CRZGate, 2, 1),
    (standard_gates.CSXGate, 2, 0),
    (standard_gates.CUGate, 2, 4),
    (standard_gates.CU1Gate, 2, 1),
    (standard_gates.CU3Gate, 2, 3),
    (standard_gates.CYGate, 2, 0),
    (standard_gates.CZGate, 2, 0),
    (standard_gates.RXXGate, 2, 1),
    (standard_gates.RYYGate, 2, 1),
    (standard_gates.RZZGate, 2, 1),
    (standard_gates.RZXGate, 2, 1),
    (standard_gates.XXMinusYYGate, 2, 2),
    (standard_gates.XXPlusYYGate, 2, 2),
    (standard_gates.ECRGate, 2, 0),
    (standard_gates.CSGate, 2, 0),
    (standard_gates.CSdgGate, 2, 0),
    (standard_gates.SwapGate, 2, 0),
    (standard_gates.iSwapGate, 2, 0),
]

def random_clifford_circuit(num_qubits, num_gates, gates="all", seed=None):
    """Generate a pseudo-random Clifford circuit.

    This function will generate a Clifford circuit by randomly selecting the chosen amount of Clifford
    gates from the set of standard gates in :mod:`qiskit.circuit.library.standard_gates`. For example:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit.circuit.random import random_clifford_circuit

       circ = random_clifford_circuit(num_qubits=2, num_gates=6)
       circ.draw(output='mpl')

    Args:
        num_qubits (int): number of quantum wires.
        num_gates (int): number of gates in the circuit.
        gates (list[str]): optional list of Clifford gate names to randomly sample from.
            If ``"all"`` (default), use all Clifford gates in the standard library.
        seed (int | np.random.Generator): sets random seed/generator (optional).

    Returns:
        QuantumCircuit: constructed circuit
    """

    gates_1q = list(set(_BASIS_1Q.keys()) - {"v", "w", "id", "iden", "sinv"})
    gates_2q = list(_BASIS_2Q.keys())

    if gates == "all":
        if num_qubits == 1:
            gates = gates_1q
        else:
            gates = gates_1q + gates_2q

    instructions = {
        "i": (standard_gates.IGate(), 1),
        "x": (standard_gates.XGate(), 1),
        "y": (standard_gates.YGate(), 1),
        "z": (standard_gates.ZGate(), 1),
        "h": (standard_gates.HGate(), 1),
        "s": (standard_gates.SGate(), 1),
        "sdg": (standard_gates.SdgGate(), 1),
        "sx": (standard_gates.SXGate(), 1),
        "sxdg": (standard_gates.SXdgGate(), 1),
        "cx": (standard_gates.CXGate(), 2),
        "cy": (standard_gates.CYGate(), 2),
        "cz": (standard_gates.CZGate(), 2),
        "swap": (standard_gates.SwapGate(), 2),
        "iswap": (standard_gates.iSwapGate(), 2),
        "ecr": (standard_gates.ECRGate(), 2),
        "dcx": (standard_gates.DCXGate(), 2),
    }

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    samples = rng.choice(gates, num_gates)

    circ = QuantumCircuit(num_qubits)

    for name in samples:
        gate, nqargs = instructions[name]
        qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
        circ.append(gate, qargs, copy=False)

    return circ

def get_random_pauli_string(num_qubits):

    paulis = ['X','Y','Z','I']
    initial_pauli_string = ''
    for i in range(num_qubits):
        pauli = random.choice(paulis)
        if pauli == 'X':
            initial_pauli_string += 'X'
        elif pauli == 'Y':
            initial_pauli_string += 'Y'
        elif pauli == 'Z':
            initial_pauli_string += 'Z'
        elif pauli == 'I':
            initial_pauli_string += 'I'

    return initial_pauli_string

def apply_initial_step(qc, num_qubits, pauli_string):

    for i in range(num_qubits):
        pauli = pauli_string[i]
        if pauli == 'X':
            qc.x(num_qubits - 1 - i)
        elif pauli == 'Y':
            qc.y(num_qubits - 1 - i)
        elif pauli == 'Z':
            qc.z(num_qubits - 1 - i)
    qc.barrier()

    return qc

def apply_final_step(qc, num_qubits, measurement_basis):

    for i in range(num_qubits):
        pauli = measurement_basis[i]
        if pauli == 'Y':
            qc.sdg(num_qubits - 1 - i)
            qc.h(num_qubits - 1 - i)
        elif pauli == 'X':
            qc.h(num_qubits - 1 - i)

    return qc

def obtain_stabilizer(num_qubits, initial_pauli_string):

    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if initial_pauli_string[i] == 'X':
            qc.x(num_qubits - 1 - i)
        elif initial_pauli_string[i] == 'Y':
            qc.y(num_qubits - 1 - i)
        elif initial_pauli_string[i] == 'Z':
            qc.z(num_qubits - 1 - i)
    
    stab = StabilizerState(qc)
    cliff = Clifford(qc)

    #stab_operator = stab.to_operator() # Calculate the stabilizer in operator form
    #print('Initial ', initial_pauli_string)
    #print('Stabilizer', stabstate)
    print('Clifford', cliff, 'labels', cliff.to_labels(mode='S'))
    stab_string = cliff.to_labels(mode='S')[0][-num_qubits:]
    print(stab_string)
    # We check that the operator is not an identity
    counter = 0
    for string in stab_string:
        if string == 'I':
            counter += 1
    if counter == num_qubits:
        print('THis is not a proper stabilizer, exiting...')
        exit()
    sign = cliff.to_labels(mode='S')[0][0]
    if sign == '-':
        stab = SparsePauliOp(stab_string, coeffs=[-1])
    elif sign == '+':
        stab = SparsePauliOp(stab_string, coeffs=[1])
    else:
        print('This is not a proper sign, exiting')
        exit()

    return stab

def obtain_observable(num_qubits):

    #observables = []
    #for i in range (num_qubits):
    #    pauli_string = ''
    #    for j in range(num_qubits):
    #        if i == j:
    #            pauli_string += 'Z'
    #        else:
    #            pauli_string += 'I'
    #    observable = SparsePauliOp(pauli_string)
    #    observables.append(observable)
    pauli_string = ''
    for i in range(num_qubits):
        pauli_string += 'Z'
    observable = SparsePauliOp(pauli_string)

    return observable

def add_random_clifford(qc, num_qubits, depth, pass_manager):

    obtained_depth = 0
    while_counter = 0
    while obtained_depth != depth:
        while_counter += 1
        #print('Trial {}'.format(while_counter))
        #rc = random_clifford(num_qubits=num_qubits).to_circuit()
        number_of_gates = random.randrange(100)
        rc = random_clifford_circuit(num_qubits, number_of_gates)
        #print(rc.depth())
        if rc.depth() == depth:
            qc = qc.compose(rc)#, front=True)
            obtained_depth = depth 

    #isa_circuit = pass_manager.run(qc)

    return qc #isa_circuit

def calculate_observable(qc, backend, pass_manager, sign):

    # MAPPING
    isa_circuit = pass_manager.run(qc)
    isa_observable = observable.apply_layout(isa_circuit.layout)
    pub = [isa_circuit, isa_observable]
    estimator = Estimator(mode = backend)#, options={"default_shots": 1024, "resilience_level":2})
    job = estimator.run([pub])
    print(job.result()[0].data.evs*sign)

    return job.result()[0].data.evs*sign

def calculate_distribution(qc, backend, pass_manager):

    qc.measure_all()
    isa_circuit = pass_manager.run(qc)
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit])
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    print(counts)

    return counts

def obtain_measurement_basis(u, stab):

    udagger = u.adjoint() # Calculate the adjoint of the previous operator.
    pauli_gate = SparsePauliOp.from_operator(u.compose(stab, front=True).compose(udagger, front=True)) # calculate U*P*Udagger
    #pauli_gate = SparsePauliOp.from_operator(udagger.compose(stab).compose(u)) # calculate U*P*Udager
    print('final ', pauli_gate.to_list())
    measurement_basis = pauli_gate.to_list()[0][0]
    real_part = pauli_gate.to_list()[0][1].real
    imag_part = pauli_gate.to_list()[0][1].imag
    print('Pauli ', measurement_basis, 'real part of coeff ', real_part, 'imaginary part of coeff ', imag_part)
    if real_part > 0:
        sign = 1
    elif real_part < 0:
        sign = -1
    else: 
        print('Imaginary coefficient, exiting')
        exit()

    return measurement_basis, sign


backend = FakeTorino()
pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)

num_qubits = 2 # Number of qubits
depth_sequence = [2,4,6,8,10] # Depth list
unitary_clifford_amount_sequence = [5,5,5,5,5] # Number of unitary cliffords for each depth
k = 1 # Number of times each unitary clifford is executed. Not implemented
f = [] # Here we store the values to plot and fit later
observable = obtain_observable(num_qubits) # The ...ZZZZZ.... observable to measure in al circuits

depth_counter = 0
for depth in depth_sequence: # Loop for different depths
    print('working on depth {}'.format(depth))
    f_average = 0
    unitary_clifford_amount = unitary_clifford_amount_sequence[depth_counter]
    initial_pauli_string = get_random_pauli_string(num_qubits) # Generate random pauli operator
    stab = obtain_stabilizer(num_qubits, initial_pauli_string) # Calculate the stabilizer of the random state
    for i in range(unitary_clifford_amount): # Loop for circuits with different cliffords of same depth
        qc = QuantumCircuit(num_qubits) # Create quantum circuit
        qc = apply_initial_step(qc, num_qubits, initial_pauli_string) # Apply initial random pauli operator
        qc = add_random_clifford(qc, num_qubits, depth, pass_manager) # Add random clifford
        qc.barrier() # Add barrier to separate random clifford from measurement paulis
        u = Operator.from_circuit(qc, layout=qc.layout) # Calculate the operator associated to the inital layer + random clifford
        measurement_basis, sign = obtain_measurement_basis(u, stab) # Obatin the measurement basis by calculating UPU^dagger
        qc = apply_final_step(qc, num_qubits, measurement_basis) # Layer for changing measurement basis https://quantumcomputing.stackexchange.com/questions/13605/how-to-measure-in-another-basis
        print(qc)
        observable_value = calculate_observable(qc, backend, pass_manager, sign) # calculate the observable, measurement in computational basis
        distribution = calculate_distribution(qc, backend, pass_manager) # Calculate the distribution
        f_average += observable_value
    f.append(f_average/unitary_clifford_amount) # Calculate the average value
    depth_counter += 1

plt.plot(depth_sequence, f)
plt.show()