function [E_a, Cv_a, M_a]=Onsager(temps,J)
% Calcula la solución analítica del modelo de Ising bidimensional.

Tc = 2*J/log(1+sqrt(2));
T = temps;

b = 1./T;
K = 2*sinh(2*b*J) ./ (cosh(2*b*J)).^2;
dKdb = 4*J./cosh(2*b*J).*(1-2*J*tanh(2*b*J).^2);

for i=1:length(T)
    dfdb = @(x) -K(i)*sin(x).^2./( sqrt(1-K(i)^2.*sin(x).^2)+(1-K(i)^2.*sin(x).^2) ) * dKdb(i);
    Integral_dfdb(i) = integral(dfdb,0,pi)/2/pi;
end

E_a = -(tanh(2*b*J)*2*J + Integral_dfdb);
Cv_a =[0 E_a(3:end)-E_a(1:end-2) 0]/(T(3)-T(1));
Cv_a(1)=Cv_a(2); 
Cv_a(end)=Cv_a(end-1); 

T_m=T(1:find(T==Tc));
M_a=(1-(sinh(2./T_m)).^(-4)).^(1/8);
M_a=[M_a zeros(1, length(T(find(T==Tc)+1:end)) )];
end