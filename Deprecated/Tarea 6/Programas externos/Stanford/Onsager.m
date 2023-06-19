function [F_anl, E_anl, dEdT]=Onsager(temps)

% Plot Onsager's solution of 2D Ising model
% ME 346A  Winter 2011
% Wei Cai  caiwei@stanford.edu

J = 1;
kBTc = 2*J/log(1+sqrt(2));
%kBT = [0.1:0.001:3];
kBT = temps;

beta = 1./kBT;
K = 2*sinh(2*beta*J) ./ (cosh(2*beta*J)).^2;
dKdbeta = 4*J./cosh(2*beta*J).*(1-2*J*tanh(2*beta*J).^2);

fint = zeros(size(kBT));
dfintbeta = zeros(size(kBT));
for i=1:length(kBT)
    f = @(x) log((1+sqrt(1-K(i)^2.*sin(x).^2)));
    fint(i) = quadl(f,0,pi,1e-8)/2/pi;
    dfdbeta = @(x) -K(i)*sin(x).^2./( sqrt(1-K(i)^2.*sin(x).^2)+(1-K(i)^2.*sin(x).^2) ) * dKdbeta(i);
    dfintdbeta(i) = quadl(dfdbeta,0,pi,1e-8)/2/pi;
    waitbar(i/length(kBT));
end
close(h) 
    
F_anl = -kBT.*( log(sqrt(2)*cosh(2*beta*J)) + fint);
E_anl = -(tanh(2*beta*J)*2*J + dfintdbeta);

% compute dEdT from numerical difference
dEdT = [0, E_anl(3:end)-E_anl(1:end-2), 0]/(kBT(3)-kBT(1));
dEdT(1) = dEdT(2);
dEdT(end) = dEdT(end-1);

[tmp,indc] = min(abs(kBT-kBTc));
end