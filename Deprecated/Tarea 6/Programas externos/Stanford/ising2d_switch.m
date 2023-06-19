%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%File: ising2d_switch.m
%
%Purpose: Adiabatic Switching simulation of 2D Ising model
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

kT=1.0; %reference temperature

%n = 40;
%n = 80;
n = 100;

N = n*n;  %total number of spins

disp(sprintf('parameters: J=%f kT=%f number of spins = %d x %d',J,kT,n,n));

F_kT_1 = -2.0003; %analytic solution (see plot_Onsager_solution.m)
F_kT_4 = -3.0364; %analytic solution (see plot_Onsager_solution.m)

savefreq = N;
%Niter =  savefreq*200;  %a very short test case
%Niter =  savefreq*1000;
Niter =  savefreq*5000;
plotfreq = savefreq*10;

switch_func = inline('1-0.75*s','s'); %lambda = 1 to 1/4 as s=iter/Niter goes from 0 to 1

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
dUdlam_data = zeros(1,Niter/savefreq);
Wdata = zeros(1,Niter/savefreq);


figure(1);
%plot initial configuration
%spy(S+1);

idx=ceil(n*rand(Niter,1));  %pre-generate all random numbers
jdx=ceil(n*rand(Niter,1));  % that will be used in the
eps=rand(Niter,1);          % simulation

%do not precompute flip rate because it changes with lambda

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
Wdyn = 0; %dynamical work

lam_array = switch_func([0:Niter]/Niter);
for iter=1:Niter,
    lambda = lam_array(iter+1);
    dlam   = lam_array(iter+1)-lam_array(iter);

    i=idx(iter); %randomly select a spin (i,j)
    j=jdx(iter); %
    il=IL(i);
    ir=IR(i);
    jl=IL(j);
    jr=IR(j);
    dss_over_2 = S(i,j)*(S(i,jl)+S(i,jr)+S(il,j)+S(ir,j)); %compute change of energy / J
    acc=1;
    if(dss_over_2>0)
        acc_prob = exp(-dss_over_2*2*lambda*J/kT);
        if(eps(iter)>acc_prob) % compute acceptance probability at every step
            acc=0;
        end
    end
    if(acc)
        S(i,j)=-S(i,j); %trial accepted
        SS = SS - dss_over_2*2; %update sum_<i,j> s_i s_j
    end
    U = -lambda*J*SS; %potential energy
    dUdlam = -J*SS;
    Wdyn = Wdyn + dUdlam*dlam; %accumulate dynamic work
    
    if(mod(iter,plotfreq)==0)
        figure(1);
        spy(S+1,'k'); %visualization
        drawnow;
        disp(sprintf('iter=%d/%d',iter,Niter));
    end
    if(mod(iter,savefreq)==1)
        %data(:,(iter-1)/savefreq+1)=reshape(S,n*n,1);
        Mdata((iter-1)/savefreq+1)=sum(sum(S));
        Udata((iter-1)/savefreq+1)=U;
        dUdlam_data((iter-1)/savefreq+1)=dUdlam;
        Wdata((iter-1)/savefreq+1) = Wdyn;
    end
end
  
figure(1); 
spy(S+1); %visualization

figure(2);
plot(Udata / N);
xlabel('time step');
ylabel('U / N');

figure(3);
plot(dUdlam_data);
xlabel('time step');
ylabel('dU / d\lambda');

figure(4);
plot(Wdata);
xlabel('time step');
ylabel('W_{dyn}');

kTnew = kT./lam_array(1:savefreq:end-1);

%F0 = U0; %approximation at low temperature
F0 = F_kT_1 * N; %analytic solution at kT = 1

Fnew = (F0 + Wdata)./lam_array(1:savefreq:end-1);
figure(5);
plot(kTnew, Fnew/N);
xlabel('T_{new}');
ylabel('F / N');

disp(sprintf('Fnew(kT=4) / N = %g (numeric)',Fnew(end)/N));
disp(sprintf('Fnew(kT=4) / N = %g (analytic)',F_kT_4));