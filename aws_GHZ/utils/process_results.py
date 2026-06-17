import os
import json
import re
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import qiskit.qasm3

file_path = ".\\results\\job_4184_executor_8654_output.json" 
base_output_dir = "processed_results"

COMPACT_LAYOUT_QUIL = True  

def quil_to_qiskit_circuit(quil_str, compact=True):
    lines = [line.strip() for line in quil_str.split('\n') if line.strip()]
    physical_qubits = set()
    num_cbits = 0
    
    for line in lines:
        if line.startswith("DECLARE"):
            match = re.search(r"BIT\[(\d+)\]", line)
            if match: num_cbits = int(match.group(1))
        elif line.startswith("RZ") or line.startswith("RX"):
            match = re.match(r"R[ZX]\((.*?)\)\s+(\d+)", line)
            if match: physical_qubits.add(int(match.group(2)))
        elif line.startswith("CZ"):
            match = re.match(r"CZ\s+(\d+)\s+(\d+)", line)
            if match:
                physical_qubits.add(int(match.group(1)))
                physical_qubits.add(int(match.group(2)))
        elif line.startswith("MEASURE"):
            match = re.match(r"MEASURE\s+(\d+)", line)
            if match: physical_qubits.add(int(match.group(1)))
                
    sorted_physical = sorted(list(physical_qubits))
    if compact:
        qubit_map = {phys: idx for idx, phys in enumerate(sorted_physical)}
        num_qubits = len(sorted_physical) if sorted_physical else 1
    else:
        qubit_map = {phys: phys for phys in sorted_physical}
        num_qubits = (max(sorted_physical) + 1) if sorted_physical else 1
        
    if num_cbits == 0: num_cbits = num_qubits
    qc = QuantumCircuit(num_qubits, num_cbits)
    qc.metadata = {"original_qubits": sorted_physical, "layout_mapping": qubit_map}
    
    for line in lines:
        if line.startswith("PRAGMA") or line.startswith("DECLARE"): continue
        if line.startswith("RZ"):
            match = re.match(r"RZ\((.*?)\)\s+(\d+)", line)
            if match: qc.rz(float(match.group(1)), qubit_map[int(match.group(2))])
        elif line.startswith("RX"):
            match = re.match(r"RX\((.*?)\)\s+(\d+)", line)
            if match: qc.rx(float(match.group(1)), qubit_map[int(match.group(2))])
        elif line.startswith("CZ"):
            match = re.match(r"CZ\s+(\d+)\s+(\d+)", line)
            if match: qc.cz(qubit_map[int(match.group(1))], qubit_map[int(match.group(2))])
        elif line.startswith("MEASURE"):
            match = re.match(r"MEASURE\s+(\d+)\s+ro\[(\d+)\]", line)
            if match: qc.measure(qubit_map[int(match.group(1))], int(match.group(2)))
    return qc

def sanitize_qasm3_dialect(qasm_str):
    if 'include "stdgates.inc";' not in qasm_str:
        qasm_str = re.sub(
            r'(OPENQASM\s+3(?:\.0)?\s*;)', 
            r'\1\ninclude "stdgates.inc";', 
            qasm_str, 
            flags=re.IGNORECASE
        )

    qasm_str = re.sub(r'\bccnot\b', 'ccx', qasm_str)
    qasm_str = re.sub(r'\bcnot\b', 'cx', qasm_str)
    
    return qasm_str

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

backend_arn = data.get("backend", "arn:aws:braket:::device/qpu/unknown/unknown")
try:
    parts = backend_arn.split("/")
    provider = parts[-2].lower()
    backend_name = parts[-1].lower()
except IndexError:
    provider, backend_name = "provider", "backend"

date_str = data.get("date", datetime.now().strftime("%Y%m%d_%H%M%S"))

os.makedirs(base_output_dir, exist_ok=True)

# ==========================================
# Save calibration/properties
# ==========================================
properties = {}
for k, v in data.items():
    if "properties" in k.lower() or "capabilities" in k.lower():
        properties = v
        break
if not properties:
    properties = {k: v for k, v in data.items() if not k.endswith("_qubits") and k != "backend"}

prop_base_name = f"qubit_properties_{provider}_{backend_name}_{date_str}"
with open(os.path.join(base_output_dir, f"{prop_base_name}.json"), "w", encoding="utf-8") as f:
    f.write(properties)
with open(os.path.join(base_output_dir, f"{prop_base_name}.pkl"), "wb") as f:
    pickle.dump(properties, f)

# ==========================================
# Plot circuits
# ==========================================

circuits_dir_path = os.path.join(base_output_dir, f"circuits_{provider}_{backend_name}_{date_str}")
os.makedirs(circuits_dir_path, exist_ok=True)

contador_ok = 0
contador_error = 0

for key, value in data.items():
    if "_qubits" in key and isinstance(value, dict):
        metadata_list = value.get("metadata", [])
        
        for idx, item in enumerate(metadata_list):
            task_id = item.get("task_id", "").split("/")[-1] or f"id_{idx}"
            
            untranspiled_qasm = item.get("untranspiled_circuit")
            if untranspiled_qasm:
                prefix = f"untranspiled_{key}_{task_id}"
                try:
                    untranspiled_qasm = sanitize_qasm3_dialect(untranspiled_qasm)
                    
                    qc_untranspiled = qiskit.qasm3.loads(untranspiled_qasm)
                    
                    with open(os.path.join(circuits_dir_path, f"{prefix}.qasm"), "w", encoding="utf-8") as f_qasm:
                        qiskit.qasm3.dump(qc_untranspiled, f_qasm)
                    
                    qc_untranspiled.draw('mpl', filename=os.path.join(circuits_dir_path, f"{prefix}.png"))
                    plt.close()

                    with open(os.path.join(circuits_dir_path, f"{prefix}.pkl"), 'wb') as f_pkl:
                        pickle.dump(qc_untranspiled, f_pkl)
                        
                    print(f" -> [OK] {prefix} (QASM, PNG, PKL)")
                    contador_ok += 1
                except Exception as e:
                    print(f" -> [ERROR] {prefix}: {e}")
                    contador_error += 1

            compiled_str = item.get("circuit")
            if compiled_str:
                prefix = f"transpiled_{key}_{task_id}"
                try:
                    if "PRAGMA" in compiled_str or "DECLARE" in compiled_str:
                        qc_compiled = quil_to_qiskit_circuit(compiled_str, compact=COMPACT_LAYOUT_QUIL)
                    else:
                        compiled_str = sanitize_qasm3_dialect(compiled_str)
                        qc_compiled = qiskit.qasm3.loads(compiled_str)
                    
                    with open(os.path.join(circuits_dir_path, f"{prefix}.qasm"), "w", encoding="utf-8") as f_qasm:
                        qiskit.qasm3.dump(qc_compiled, f_qasm)
                    
                    qc_compiled.draw('mpl', filename=os.path.join(circuits_dir_path, f"{prefix}.png"))
                    plt.close()
                    
                    with open(os.path.join(circuits_dir_path, f"{prefix}.pkl"), 'wb') as f_pkl:
                        pickle.dump(qc_compiled, f_pkl)
                        
                    contador_ok += 1
                except Exception as e:
                    print(f" -> [ERROR] {prefix}: {e}")
                    contador_error += 1

# ==========================================
# TAREA 4: Separar resultados numéricos por tamaño de qubits
# ==========================================
for key, value in data.items():
    if "_qubits" in key and isinstance(value, dict):
        try:
            num_qubits = int(key.split("_")[0])
        except ValueError:
            continue
        
        all_counts = []
        if "metadata" in value and isinstance(value["metadata"], list):
            for meta in value["metadata"]:
                if "measurement_counts" in meta:
                    all_counts.append(meta["measurement_counts"])
        if not all_counts and "measurement_counts" in value:
            all_counts.append(value["measurement_counts"])
            
        results_structure = {
            "backend": backend_arn,
            "numero_qubits_inicial": num_qubits,
            "qubits": value.get("qubits", num_qubits),
            "epsilon": value.get("epsilon", 0.5),
            "delta": value.get("delta", 0.5),
            "all_counts": all_counts,
            "P_C": {
                "P": value.get("P", 0.0),
                "C": value.get("C", 0.0)
            },
            "fidelity": value.get("fidelity_estimate", value.get("fidelity", 0.0))
        }
        
        res_filename = f"results_{provider}_{backend_name}_{num_qubits}q_{date_str}.json"
        with open(os.path.join(base_output_dir, res_filename), "w", encoding="utf-8") as f:
            json.dump(results_structure, f, indent=4)