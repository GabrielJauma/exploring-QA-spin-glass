numSpinsPerDim = 2^3;
probSpinUp = 0.5;
J = 1;

% Temperatures to sample
numTemps = 10;
kTc = 2*J / log(1+sqrt(2)); % Curie temperature
kT = linspace(0, 2*kTc, numTemps);

% Preallocate to store results
Emean = zeros(size(kT));
Mmean = zeros(size(kT));

% Replace 'for' with 'parfor' to run in parallel with Parallel Computing Toolbox.
for tempIndex = 1 : numTemps
    tempIndex 
    'of'
     numTemps
    spin = initSpins(numSpinsPerDim, probSpinUp);
    spin = metropolis(spin, kT(tempIndex), J);
    Emean(tempIndex) = energyIsing(spin, J);
    Mmean(tempIndex) = magnetizationIsing(spin);
end

plot(kT,Emean,'.')
