import random 
#from typing import override
from typing_extensions import override # With older python version
import numpy as np
import json
import sys

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_clifford
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes import (
    #Optimize1qGatesDecomposition,
    Optimize1qGates,
    #CXCancellation,
    CommutativeCancellation,
    RemoveResetInZeroState,
    RemoveFinalReset,
    RemoveIdentityEquivalent,
    OptimizeSwapBeforeMeasure,
    RemoveFinalMeasurements,
    BarrierBeforeFinalMeasurements,
    RemoveBarriers,
    # NOTE: we intentionally *do not* import RemoveDiagonalGatesBeforeMeasure
)
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from birb_test import BiRBTest

class BiRBTestCP(BiRBTest):
    """
    Class for testing BiRB on circuits composed of layers where only a
    percentage of each layer consists of a Clifford circuit. To construct the
    layer, we begin by generating numerous random Clifford unitaries and
    transpiling them to a target backend. For each transpiled circuit, we
    extract a specified percentage of the circuit and compute a metric (e.g.,
    two-qubit gate depth or count). Using this metric, we perform a binary
    search to determine the appropriate percentage of a Clifford circuit
    (composed of one- and two-qubit gates) that, after transpilation, yields a
    similar metric. The goal is to match the metric (such as two-qubit gate
    depth) between this adapted approach and the original transpiled circuits.    

    Args:
        Same as BiRBTest, except for `percent`, which specifies the percentage
        of the Clifford layer used in each test circuit.
    """

    def __init__(self, qubits, depths, sim_type, execution_mode, backend_name, account_name,
                 circuits_per_depth=int(1e5), shots_per_circuit=int(1e5),
                 percent=0.5):

        """
            Before constructing any layer, we call a function to compute the
            appropriate circuit percentage to use.
        """

        super().__init__(qubits, depths, sim_type, execution_mode, backend_name, account_name,
                         circuits_per_depth, shots_per_circuit)

        self.percent = percent
        self.tolerance = 0.1

        self.adapted_percent = -1
        self.depth_2q_gate = -1
        self.quantity_2q_gate = -1

    def get2qDepth(self):
        assert(self.depth_2q_gate >= 0)
        return self.depth_2q_gate
    
    def get2qQuantity(self):
        assert(self.quantity_2q_gate >= 0)
        return self.quantity_2q_gate

    def getAdaptedPercent(self):
        assert(self.adapted_percent >= 0)
        return self.adapted_percent

    def _findPercent(self, type, num_tries, tolerance):

        """
            Determine the adapted percentage such that building a Clifford unitary circuit,
            extracting the adapted percentage, and then transpiling it, results in circuit
            characteristics similar to those obtained by transpiling the full unitary first
            and then extracting the percentage.

            Args:
                type (str): Specifies the metric used for comparison. Accepted values are:
                    - "depth": Compares the depth of two-qubit gates.
                    - "quantity": Compares the number of two-qubit gates.
                num_tries (int): Number of random circuits generated to estimate the metric.
                tolerance (float): Acceptable error margin for the metric comparison.

            Return:
                mid_percent (float): the adapted percent. 
        """

        low_percent, mid_percent, up_percent = 0, 0.5, 1

        # Depth and number of two qubit gates of the circuit that transform the
        # unitary to circuit, get a percent, and then transplie
        metrics_transpile_slice = {"Gate depth": 0.0, "Gate quantity": 0.0}

        # Depth and number of two qubit gates of the circuit that first get the
        # circuit of the unitary and then take teh percent
        metrics_slice_transpile = {"Gate depth": 0.0, "Gate quantity": 0.0} 

        if(type == "depth"):
            metric_name = "Gate depth"
        elif(type == "quantity"):
            metric_name = "Gate quantity"
        else:
            raise Exception (f"Parameter {type} not valid")

        # Get the depth and number of gates of the transpiled circuit
        (
            metrics_transpile_slice["Gate depth"],
            metrics_transpile_slice["Gate quantity"]
        ) = self._getDepthCircuit("transpile_slice", num_tries, self.percent)

        console = Console()

        # Binary search
        while(abs(metrics_transpile_slice[metric_name] -
            metrics_slice_transpile[metric_name]) >= tolerance):

            mid_percent = (low_percent + up_percent)/2

            console.print(Panel.fit(
                f"[bold blue]🔎 Binary search step[/bold blue]\nLow:"
                f"{low_percent}   Mid: {mid_percent}   Up: {up_percent}",
                border_style="cyan",
                title="Binary Search"
            ))

            # Get the depth and number of two qubit gates of the circuit
            # results of slicing and then transpiling
            (
                metrics_slice_transpile["Gate depth"],
                metrics_slice_transpile["Gate quantity"] 
            ) = self._getDepthCircuit("slice_transpile", num_tries, mid_percent)


            table = Table(title="📈 Metric results", border_style="green")
            table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
            table.add_column("Goal", justify="center")
            table.add_column("Obtained", justify="center")

            table.add_row("▶ "+metric_name,
                          str(metrics_transpile_slice[metric_name]),
                          str(metrics_slice_transpile[metric_name]))

            table.add_row("Difference", str(self.tolerance),
                          str(abs(metrics_transpile_slice[metric_name] -
                          metrics_slice_transpile[metric_name])))

            if(metric_name == "Gate depth"): other_metric = "Gate quantity"
            else: other_metric = "Gate depth"
            table.add_row(other_metric,
                          str(metrics_transpile_slice[other_metric]),
                          str(metrics_slice_transpile[other_metric]))

            console.print(table)

            
            if(metrics_slice_transpile[metric_name] <
                   metrics_transpile_slice[metric_name]):

                low_percent = mid_percent
            else:

                up_percent = mid_percent

        return mid_percent, metrics_slice_transpile["Gate depth"], metrics_slice_transpile["Gate quantity"]

    def _getDepthCircuit(self, method, num_tries, percent):
        """
            Compute the depth of two-qubit gates and the number of such gates within a specified
            percentage of random quantum circuits. The circuits are constructed using different 
            strategies, as specified by the `method` parameter.

            Args:
                method (str): Strategy used to build and process the circuit for metric computation.
                    Accepted values are:
                        - "transpile_slice": Construct the unitary, transpile it to the target backend,
                          and then extract the specified percentage of the resulting circuit.
                        - "slice_transpile": Construct the unitary, decompose it into one and two qubit
                          gates, extract the specified percentage, and then transpile the resulting circuit.

                num_tries (int): Number of random circuits to generate in order to estimate the metrics.

                percent (float): Fraction of the circuit (expressed as a percentage) to consider
                    when computing the metrics.

            Returns:
                - The average depth of the two-qubit gates across all generated circuits.
                - The average number of two-qubit gates within the specified circuit percentage.
        """

        mean_gates, mean_depth = 0, 0

        console = Console()

        console.print(f"[bold magenta]⚙️  Computing {method} "
                       "circuit metrics...[/bold magenta]")


        with Progress(TextColumn("[bold blue]Processing[/bold blue]"),
                      BarColumn(),
                      TextColumn("([bold]{task.completed}[/] / {task.total})"),
                      TimeElapsedColumn(),
                      ) as progress:

            task = progress.add_task("[green]Evaluando circuitos...", total=1000)
            for _ in range(1, num_tries + 1):
                progress.update(task, advance=1)
                unitary = random_clifford(self.qubits)

                if(method == "transpile_slice"):
                    qc = QuantumCircuit(self.qubits)
                    qc.append(unitary, range(self.qubits))
                    pm3 = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
                    pm3.post_optimization = PassManager([
                                        OptimizeSwapBeforeMeasure(),
                                        RemoveBarriers(),                 # harmless clean-up
                                        BarrierBeforeFinalMeasurements(), # preserves measurement structure
                                        ])
                    tranpiled_qc = pm3.run(qc)
                    circuit = self._getPercent(tranpiled_qc, percent) 

                elif method == "slice_transpile":
                    qc = unitary.to_circuit()
                    reduced_qc = self._getPercent(qc, percent)
                    pm3 = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
                    pm3.post_optimization = PassManager([
                                        OptimizeSwapBeforeMeasure(),
                                        RemoveBarriers(),                 # harmless clean-up
                                        BarrierBeforeFinalMeasurements(), # preserves measurement structure
                                        ])
                    circuit = pm3.run(reduced_qc)

                else:
                    raise Exception(f"Method {method} not valid")
             
                # Count the number of layers of two qubit gates
                mean_depth += circuit.depth(lambda instr: len(instr.qubits) > 1)

                # Count the total number of two qubit gates
                mean_gates += (lambda qc: 
                                    sum(1 for inst in qc.data if len(inst.qubits) > 1
                               ))(circuit)
             
        return mean_depth / num_tries, mean_gates / num_tries

    def _getPercent(self, qc, percent):

        """
        Takes a Clifford operator selects a portion of it corresponding to
        `percent` of its total depth. The starting point for extracting
        this subcircuit is chosen randomly.

        Args:
            qc (QuantumCircuit): Circuit to take the percent.
            percent (float): Percent of the circuit to take.

        Returns:
            qc (QuantumCircuit): A circuit composed of `self.percent` percent of the 
            depth of a Clifford circuit on `self.n` qubits.
        """

        if percent == 1.0:
            return qc.copy()

        dag = circuit_to_dag(qc)
        layers = list(dag.layers())
        total = len(layers)
        num = int(np.floor(total * percent))
        start = random.randint(0, total - num)
        end = start + num

        qc2 = QuantumCircuit(*qc.qregs, *qc.cregs, name=f"{qc.name}_sub")
        for i in range(start, end):
            for node in layers[i]['graph'].op_nodes():
                qc2.append(node.op, node.qargs, node.cargs)
        return qc2

    @override
    def _generateRandomLayer(self):

        """
        Takes a Clifford operator, decomposes it into a specific circuit, and selects 
        a portion of it corresponding to `self.adapted_percent` of its total depth. 
        The starting point for extracting this subcircuit is chosen randomly.

        Returns:
            qc (QuantumCircuit): A circuit composed of `self.adapted_percent` percent of the 
            depth of a Clifford circuit on `self.n` qubits.
        """

        clifford_circuit = random_clifford(self.qubits).to_circuit()

        if(self.percent == 1.0):
            return clifford_circuit

        return self._getPercent(clifford_circuit, self.adapted_percent)

    @override
    def prepareCircuits(self, file_prefix):

        """
        Override the parent function to calculate first the adapted percent we
        need for the circuits, and then call the parents functions to store the
        circuits

        """
        if(self.percent == 1.0):
            self.adapted_percent = 1.0
            ( 
                self.depth_2q_gate,
                self.quantity_2q_gate
            ) = self._getDepthCircuit("slice_transpile", 1000, self.percent)

        else:
            console = Console()
            panel = Panel(
                Align.center("Computing adapted percent for the circuit"),
                title="PREPROCESSING",
                border_style="green",
            )
            console.print(Align.center(panel))

            (
                self.adapted_percent, 
                self.depth_2q_gate, 
                self.quantity_2q_gate
            ) = self._findPercent("quantity", 1000, self.tolerance)

            panel = Panel.fit(
                f"[bold yellow]ADAPTED PERCENT:[/] [bold magenta]{self.adapted_percent:.3f}[/]",
                title="[bold green]✅ Result",
                border_style="bright_blue",
            )

            console.print(panel)

        datos = {
            "adapted_percent": self.adapted_percent,
            "depth_2q_gate": self.depth_2q_gate,
            "quantity_2q_gate": self.quantity_2q_gate
        }

        with open(file_prefix + f"config.json", "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)

        super().prepareCircuits(file_prefix)

    @override
    def prepareCircuits_old(self, file_prefix):

        """
        Override the parent function to calculate first the adapted percent we
        need for the circuits, and then call the parents functions to store the
        circuits

        """
        if(self.percent == 1.0):
            self.adapted_percent = 1.0
            ( 
                self.depth_2q_gate,
                self.quantity_2q_gate
            ) = self._getDepthCircuit("slice_transpile", 1000, self.percent)

        else:
            console = Console()
            panel = Panel(
                Align.center("Computing adapted percent for the circuit"),
                title="PREPROCESSING",
                border_style="green",
            )
            console.print(Align.center(panel))

            (
                self.adapted_percent, 
                self.depth_2q_gate, 
                self.quantity_2q_gate
            ) = self._findPercent("quantity", 1000, self.tolerance)

            panel = Panel.fit(
                f"[bold yellow]ADAPTED PERCENT:[/] [bold magenta]{self.adapted_percent:.3f}[/]",
                title="[bold green]✅ Result",
                border_style="bright_blue",
            )

            console.print(panel)

        datos = {
            "adapted_percent": self.adapted_percent,
            "depth_2q_gate": self.depth_2q_gate,
            "quantity_2q_gate": self.quantity_2q_gate
        }

        with open(file_prefix + f"config.json", "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)

        super().prepareCircuits_old(file_prefix)

    @override
    def run(self, eps=1e-4, file_prefix=""):

        config_path = file_prefix + f"config.json"

        try:
            #with open("config.json", "r", encoding="utf-8") as f:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.adapted_percent = data["adapted_percent"]
                self.depth_2q_gate = data["depth_2q_gate"]
                self.quantity_2q_gate = data["quantity_2q_gate"]

        except Exception as e:
            print(f"Error loading file {config_path}: {e}")
            sys.exit(1)
        return super().run(eps,file_prefix)

