% In this function, we will flip every spin in lattice randomly and accept or reject the flip 
% according to probability determined by function
% 
%
%
function [gridLayer, Ms, Es, E, T, stat, index_f] = spinGlass2D_paralel_with_gridLayer(grid,T,t,index,Ms,Es, k, stat, i)
J=1;
gridLayer{1} = grid;
stat = [stat, i];
N_temp = size(grid);
N = N_temp(1);
DeltaE = zeros(N);
iniTime = clock;
index_f = index+1;
limit = (t(index) - t(index - 1))*60*3;
%limit   = t*60;  % Seconds 
myCount = 1;
while etime(clock, iniTime) < limit
	% expand the matrix 9 times so that we can apply perodical boundary conditions
	A = [grid,grid,grid;grid,grid,grid;grid,grid,grid]
	for ii = 1:N
		for jj = 1:N
			II = ii + N;
			JJ = jj + N; 
			DeltaE(ii,jj) = 2*J*(rand()*A(II,JJ-1)*A(II,JJ) + ...
					     rand()*A(II,JJ+1)*A(II,JJ) + ...
					     rand()*A(II-1,JJ)*A(II,JJ) + ...
					     rand()*A(II+1,JJ)*A(II,JJ) );
		end
	end

	p_trans = exp(-DeltaE/(k*T));
	% actually we decide if flip each spin in the matrix at the same time.
	transitions = (rand(N) < p_trans).*(rand(N)< 0.1)* -2 + 1;
	grid = grid.* transitions;
	myCount = myCount + 1;
	gridLayer{myCount} = grid;
	M = sum(sum(grid));
	E = -sum(sum(DeltaE))/2;
	Ms = [Ms, M];
	Es = [Es, E];
end

%[L, num] = bwlabel(grid == 1, 4);

%plot(track,Es,'ro');
end
