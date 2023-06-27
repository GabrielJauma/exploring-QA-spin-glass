# Exploring Quantum Annealing Architectures: A Spin Glass Perspective
This repository contains the Python code associated with the scientific publication titled "Exploring Quantum Annealing Architectures: A Spin Glass Perspective".

The project explores the domain of quantum annealing and statistical mechanics by running Markov Chain Monte Carlo (MCMC) simulations with the parallel tempering algorithm. The simulations are used to calculate several properties of the graphs used in the quantum annealers by D-Wave, such as their critical temperatures and autocorrelation times. In addition, the code can be used to analyze several types of random graphs.

## Repository Structure
This repository is structured as follows:

### Modules
This folder contains several Python scripts:

graph_generator.py to generate graph instances for statistical averages.
monte_carlo.py functions regarding Markov Chain Monte Carlo and parallel tempering.
statistical_mechanics.py functions to calculate statistical mechanics properties from the data.
pade_fits.py for performing pade fits of data and extracting properties from it.
read_data_from_cluster.py functions for reading the raw data.
figures.py for functions to help in the process of generating figures.

### Processed Data
This folder contains the processed data necessary to create the figures of the scientific paper.

### Cluster
This folder contains all the scripts and data necessary to run the simulations in a cluster.

There are also two Python scripts outside of the above folders:

read_and_process_raw_data.py is a Python script to read the raw data, check it, and generate the processed data.

generate_figures_paper.ipynb is a Jupyter notebook that uses the processed data to generate the figures for the paper.

## Dependencies
The program depends on the following Python libraries:

numpy
matplotlib
numba
mpi4py
scipy
joblib
math
itertools
pandas

## Usage

python read_and_process_raw_data.py will read the raw data, check it, and generate processed data.
Open the generate_figures_paper.ipynb Jupyter notebook to generate the figures for the paper.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
If you have any questions or feedback, feel free to open an issue on this repository.
