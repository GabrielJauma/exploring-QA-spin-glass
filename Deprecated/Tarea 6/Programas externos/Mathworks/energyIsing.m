function Emean = energyIsing(spin, J)
%ENERGYISING Mean energy per spin.
%   Emean = ENERGYISING(spin, J) returns the mean energy per spin of the
%   configuration |spin|. |spin| is a matrix of +/- 1's. |J| is a scalar.

%   Copyright 2017 The MathWorks, Inc.

sumOfNeighbors = ...
      circshift(spin, [ 0  1]) ...
    + circshift(spin, [ 0 -1]) ...
    + circshift(spin, [ 1  0]) ...
    + circshift(spin, [-1  0]);
Em = - J * spin .* sumOfNeighbors;
E  = 0.5 * sum(Em(:));
Emean = E / numel(spin);
