# Benchmarking

Test suite for quantum devices accessed via the cloud.


## BiRB and Clifford Volume Benchmarks

These two benchmarks are closely related and therefore share common code.


### Usage

The main script used to run benchmarks and visualize results is `main.py`. This
file provides two primary functionalities:

- **Run an experiment**:
  To execute an experiment, all relevant parameters must be defined in a YAML (`.yml`) file. These configuration files are stored in the `experiments_definition` directory. The structure of these files is explained later. To run an experiment, use the following command:

  ```bash
  python main.py run 'path/to/experiment_definition.yml'
  ```

  This command will run the experiment and store all resulting data in a file.

- **Show results**: 
    This option is used to generate and display plots from a
    file containing experiment data. To visualize the results of an experiment, use
    the following command:

    ```bash
      python main.py show 'path/to/experiment_result'
    ```


### Experiment definitions
To run the BiRB or Clifford Volume benchmarks, specific parameters must be
defined in a `.yml` file. This configuration file should follow the structure
outlined below:

```yml
config:
  user: string
  simulation_type: string
  output_path: string

experiments:
  - name: experiment_name 
    params:
      backend: string
      qubits: integer
      depths: list of integers
      circuits_per_depth: integer
      shots_per_circuit: integer
      percents: list of floats

```

First, we begin with the general configuration shared by all experiments. This
section includes the following parameters:

- **user**: A string specifying the IBM Quantum account username. The
corresponding credentials must be available in the environment where the
command is executed.
- **simulation_type**: Specifies the type of device used to run the
experiments. The available options are:
  - `"fake"`: Uses simulators that emulate the noise of a real quantum device.
  - `"aer"`: Uses Qiskit Aer simulators.
  - `"noiseless"`: Similar to `"fake"` but without noise.
  - `"real"`: *(To be implemented)*.
- **output_path**: The path to a directory where the experiment results will be stored.

After the general configuration, any number of experiments can be defined. Each
experiment will generate a separate output file containing its results. The
following parameters must be specified for each experiment:

- **name**: A name or identifier for the experiment.
- **backend**: The name of the IBM Quantum backend (real or simulated) on which
the experiment will run.
- **qubits**: The number of qubits to be used in the experiment.
- **depths**: A sorted list (in increasing order) specifying the circuit depths
(i.e., the number of layers) to test.
- **circuits_per_depth**: The number of random Clifford circuits to run for
each specified depth.
- **shots_per_circuit**: The number of measurement shots for each circuit
execution.
- **percents**: A list of percentages used to configure the layers within
circuits. For each value in the list, circuits will be run at various depths
with layers configured to that percentage.

An example of this definition file is:

```yml
config:
  user: "david"
  simulation_type: "fake"
  output_path: "experiments_results"

experiments:
  - name: ibm_fake_torino_3q
    params:
      backend: "fake_torino"
      qubits: 3
      depths: [1, 2, 4, 6, 17, 36, 65, 100, 150, 220]
      circuits_per_depth: 100
      shots_per_circuit: 10000
      percents: [0.5, 0.7, 1.0]

  - name: ibm_fake_torino_4q
    params:
      backend: "fake_torino"
      qubits: 4
      depths: [1, 2, 4, 6, 17, 36, 65, 100, 150, 220]
      circuits_per_depth: 100
      shots_per_circuit: 10000
      percents: [0.5, 0.7, 1.0]
```

#### Clifford Volume

To define a Clifford Volume experiment, configure all the required parameters
as desired. However, the `depths` list must be set to `[1]`, as the Clifford
Volume metric is computed using only depth-1 circuits.

#### BiRB

To define a BiRB experiment, set all the parameters as previously described,
without any specific restrictions. However, if the data from the BiRB
experiment is also intended to be used for computing Clifford Volume, the
`depths` list must include the value `1`.



### Obtained data
After running one experiment definition `yml` file there will be one `json`
file for each experiment definition in the `yml` file that contains for each
percent all the polarization of the channel for each circuit. With this data
one can calculate the following things:

- The entanglement infidelity of the channel of a circuit that is the percent
of a clifford unitary free of SPAM errors. This is the main propuse of the BiRB
benchmark. 

- The entanglement infidelity of the channel of a circuit that is the percent
of a clifford unitary including SPAM errors. This is the main propuse of the
Clifford Volume benchmark. 

### Obtained Data

After executing an experiment defined in a `.yml` file, a separate `.json` file
is generated for each experiment entry. Each of these files contains, for every
specified percentage, the polarization values of the quantum channel at the
time each circuit is executed.

This data can be used to compute the following metrics:

- **Entanglement infidelity excluding SPAM errors**: This represents the
fraction of Clifford unitaries that are free from state preparation and
measurement (SPAM) errors. It is the primary objective of the BiRB benchmark.

- **Entanglement infidelity including SPAM errors**: This represents the
fraction of Clifford unitaries affected by SPAM errors. It is the primary focus
of the Clifford Volume benchmark.
 


### Structure 
The code and the results of this benchmark is inside the folder `birb_bechmark`
.This folder contains a project designed to execute the BiRB and Clifford
Volume over varying percentages of Clifford circuits. The project is organized
as follows:

- `birb_test`: Directory containing the implementation of the classes.

- `utils`: Directory containing utility functions used by the main script.

- `main.py`: Primary script with two executable functions:
    - run: Executes a set of experiments based on the parameters defined in a
            `yml` configuration file.

    - show: Loads the results of an experiment from a `json` file, generates the
    corresponding graphical outputs, and saves them.

- `experiment_results`: Directory that stores data generated from executed
experiments.

- `experiments_definition`: Directory containing the parameter configurations
for each experiment, stored in `yml` format.

- `images_results`: Directory containing graphical outputs generated from the
experiments.

- `requirements.txt`: Specifies the versions of the packages required to run
the code.
