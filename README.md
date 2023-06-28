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
To run a simulation on a computing cluster where SLURM is available you should go to the Cluster folder and then run the following command:

bash multiple_jobs_drago.sh [type of simulation] '[name_of_queue]' [graph_name] [probability_distribution_name] [Lowest value of temperature range] [Highest value of temperature range] '[size]' '[Max number of Monte Carlo Sweeps]' '[Number of jobs]'
[type of simulation] must be binned or fast. Binned outputs all of the variables discussed in the paper. Fast only outputs the variables necessary to calculate the binder cumulant.
[name of queue] is the name of the queue where you want to send the jobs
[graph name] is the name of the graph that you want to simulate. The list of available graphs is in the Python file called graph_generator.py.
[probability distribution name] must be gaussian_EA (the interactions are chosen from a gaussian pdf with 0 mean and 1 variance) or binary (binary distribution of +-1)
[Lowest value of temperature range] and [Highest value of temperature range] are self-descriptive variables. They refer to a file located in the "temperature_distributions" folder inside the Cluster folder. 
The name of this file must be [graph_name]_[distribution_name],n=[size],T=[Lowest value of temperature range]_[Highest value of temperature range].dat, and it must only contain a list fo floats representing the list of temperatures that you want to simulate in the parallel tempering algorithm.
[size] is the size of the system
[Max number of Monte Carlo Sweeps] is the max number of Monte Carlo Sweeps that the algorithm will run to thermalise the system. Then, it will run that number of MCS again to calculate thermal averages.
[Number of jobs] is the number of SLURM jobs that you want to create. Each job will attempt to simulate 10000 diferent realizations of the same graph, stopping and storing the result as it finishes with one realization and goes to the next.

An example of usage would be:
bash multiple_jobs_drago.sh fast 'medium' random_regular_7 gaussian_EA 0.5 3.0 '800' '1280000' '30'

note that the variables between quotation marks allow to include multiple elements, for instance, to do the same simulation for different sizes:

bash multiple_jobs.sh fast 'short short medium medium long' random_regular_7 gaussian_EA 0.5 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 50'

python read_and_process_raw_data.py is used to read the raw data, check it, and generate processed data.

generate_figures_paper.ipynb is used to read the processed data and generate the figures of the paper.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
If you have any questions or feedback, feel free to open an issue on this repository.
