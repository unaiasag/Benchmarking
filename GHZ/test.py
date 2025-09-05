from utils.benchmark import run_GHZ_experiment, compute_l, sample_ghz_stabilizer, run_circuit
from utils.circuits import create_ghz_circuit
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Estimator, EstimatorOptions
import numpy as np

def transpile_circuit(circuit, backend, optimization_level=3):
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    transpiled_circuit = pass_manager.run(circuit)
    return transpiled_circuit

"""

num_qubits = 3  # Example number of qubits
ghz_circuit = create_ghz_circuit(num_qubits, mode="lineal_v2")

transpiled_circuit = transpile_circuit(ghz_circuit, backend=FakeTorino())


observables, _, fidelity = run_GHZ_experiment(
                    num_qubits=num_qubits,
                    mode=AerSimulator(),
                    transpiled_circuit=ghz_circuit,
                    epsilon=0.05,
                    delta=0.05
                )
print(f"Fidelity estimate for {num_qubits} qubits: {fidelity:.4f}")

"""
def transpile_circuit(circuit, backend, optimization_level=3):
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    transpiled_circuit = pass_manager.run(circuit)
    return transpiled_circuit

l = compute_l(epsilon=0.05, delta=0.05)  # Example values for epsilon and delta
num_qubits = 3  # Example number of qubits
m_i = 1

circuit = create_ghz_circuit(num_qubits, mode="log")
samples = sample_ghz_stabilizer(num_qubits, num_samples=l)
observables = [SparsePauliOp(p) for (p, _) in samples]
signs = np.array([sign for (_, sign) in samples])
simul = AerSimulator()
#estimator = Estimator(mode=simul, options=EstimatorOptions(default_shots=m_i))
result = run_circuit(circuit, observables, shots=m_i, mode=simul)
expected_vals = np.array([pub_result.data.evs for pub_result in result])
fidelity_estimate = np.mean(expected_vals * signs)
print(f"Estimated fidelity: {fidelity_estimate}")