%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%File: ising2d.m
%
%Purpose: Monte Carlo simulation of 2D Ising model
%
%CSRC2016 summer short course on Monte Carlo methods
%Wei Cai (caiwei@stanford.edu)
%
%Note: To reduce the amount of stored data
%      increase the number savefreq.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

J=1;     %interaction strength
%kT=3.0;   %high temperature
%kT = 2.269; %phase transition temperatue Tc~2.269
%kT=2.0;   %below transition temperature
%kT=1.5;
%kT=1.0;
kT=0.5;   %low temperature
%kT = 2.269 * 0.9;

%n = 40;
n = 80;
%n = 100;

N = n*n;  %total number of spins

disp(sprintf('parameters: J=%f kT=%f number of spins = %d x %d',J,kT,n,n));

savefreq = N;
Niter =  savefreq*200; %a very short test case
%Niter =  savefreq*1000;
plotfreq = savefreq*10;

if(exist('S'))
    if(length(S)~=n)
        clear S;
    end
end

if(~exist('S'))
 % pick one
 %S = 2*round(rand(n,n))-1; %initialize spin S(i,j) matrix (random)
 S = ones(n,n)*(-1);       %initialize spin S(i,j) matrix (all -1)
end


%data=zeros(n*n,Niter/savefreq);
Mdata = zeros(1,Niter/savefreq);
Udata = zeros(1,Niter/savefreq);

figure(1);
%plot initial configuration
%spy(S+1);

idx=ceil(n*rand(Niter,1));  %pre-generate all random numbers
jdx=ceil(n*rand(Niter,1));  % that will be used in the
eps=rand(Niter,1);          % simulation

%precompute flip rate
R=zeros(4,1);
for i=1:4,                 
    R(i) = exp(-i*2*J/kT); %precompute acceptance ratio
end
IL=[n,1:n-1];
IR=[2:n,1];

%compute initial sum_<i,j> s_i s_j
SS0 = 0;
for i=1:n,
    for j=1:n,
        SS0 = SS0 + S(i,j)*(S(i,IR(j))+S(IR(i),j));
    end
end
SS = SS0;
U0 = -J*SS0; %initial potential energy

for iter=1:Niter,
    i=idx(iter); %randomly select a spin (i,j)
    j=jdx(iter); %
    il=IL(i);
    ir=IR(i);
    jl=IL(j);
    jr=IR(j);
    dss_over_2 = S(i,j)*(S(i,jl)+S(i,jr)+S(il,j)+S(ir,j)); %compute change of energy / J
    acc=1;
    if(dss_over_2>0)
        if(eps(iter)>R(dss_over_2))
            acc=0;
        end
    end
    if(acc)
        S(i,j)=-S(i,j); %trial accepted
        SS = SS - dss_over_2*2; %update sum_<i,j> s_i s_j
    end
    U = -J*SS; %potential energy
    if(mod(iter,plotfreq)==0)
        figure(1);
        spy(S+1); %visualization
        drawnow;
        disp(sprintf('iter=%d/%d',iter,Niter));
    end
    if(mod(iter,savefreq)==1)
        %data(:,(iter-1)/savefreq+1)=reshape(S,n*n,1);
        Mdata((iter-1)/savefreq+1)=sum(sum(S));
        Udata((iter-1)/savefreq+1)=U;
    end
end
  
figure(1); 
spy(S+1); %visualization

figure(2);
plot(Udata);
xlabel('time step');
ylabel('U');