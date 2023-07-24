#!/bin/bash

# Define parameters for each simulation.
# Substitute 'binned' for 'fast' if you dont want to calculate the autocorrelation times.
simulation_parameters=(
    "
    binned 'short short medium medium long' random_regular_3 gaussian_EA 0.2 1.5 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' random_regular_3 gaussian_EA 0.2 1.5 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' random_regular_3 gaussian_EA 0.2 1.5 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' random_regular_5 gaussian_EA 0.5 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' random_regular_5 gaussian_EA 0.5 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' random_regular_5 gaussian_EA 0.5 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' random_regular_7 gaussian_EA 0.5 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' random_regular_7 gaussian_EA 0.5 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' random_regular_7 gaussian_EA 0.5 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' 1D+              gaussian_EA 0.5 2.5 '100 200 400 800 1600' '160000 320000 640000 1280000 2560000' '2 4 16 32 64' 3.0
    binned 'short short medium medium long' 1D+              gaussian_EA 0.5 2.5 '100 200 400 800 1600' '160000 320000 640000 1280000 2560000' '2 4 16 32 64' 3.0
    binned 'short short medium medium long' 1D+              gaussian_EA 0.5 2.5 '100 200 400 800 1600' '160000 320000 640000 1280000 2560000' '2 4 16 32 64' 3.0
    binned 'short short medium medium long' 1D+              gaussian_EA 1.0 2.5 '100 200 400 800 1600' '160000 320000 640000 1280000 2560000' '2 4 16 32 64' 5.0
    binned 'short short medium medium long' 1D+              gaussian_EA 1.0 2.5 '100 200 400 800 1600' '160000 320000 640000 1280000 2560000' '2 4 16 32 64' 5.0
    binned 'short short medium medium long' 1D+              gaussian_EA 1.0 2.5 '100 200 400 800 1600' '160000 320000 640000 1280000 2560000' '2 4 16 32 64' 5.0
    binned 'short short medium medium long' 1D+              gaussian_EA 1.3 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 2560000' '2 4 16 32 64' 7.0
    binned 'short short medium medium long' 1D+              gaussian_EA 1.3 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 2560000' '2 4 16 32 64' 7.0
    binned 'short short medium medium long' 1D+              gaussian_EA 1.3 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 2560000' '2 4 16 32 64' 7.0
    binned 'short short medium medium long' chimera          gaussian_EA 0.2 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' chimera          gaussian_EA 0.2 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' chimera          gaussian_EA 0.2 3.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' pegasus          gaussian_EA 0.2 4.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' pegasus          gaussian_EA 0.2 4.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' pegasus          gaussian_EA 0.2 4.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' zephyr           gaussian_EA 0.5 5.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' zephyr           gaussian_EA 0.5 5.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    binned 'short short medium medium long' zephyr           gaussian_EA 0.5 5.0 '100 200 400 800 1600' '160000 320000 640000 1280000 5120000' '2 4 16 32 64'
    "
)

# Iterate over each set of parameters
for params in "${simulation_parameters[@]}"; do
    # Run the other script with these parameters
    bash multiple_jobs_drago.sh $params
done