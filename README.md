# Benchmarking

## birb_benchmark folder
This folder contains a project designed to execute the BiRB test over varying
percentages of Clifford circuits. The project is organized as follows:

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
