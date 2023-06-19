clc, clear, close all, warning('off')
%% Máster en Física de sistemas complejos, UNED
%  Modelización y simulación de sistemas complejos
%  Tarea 2. Modelo de Ising en dos dimensiones

%  Autor: Gabriel Jaumà Gómez, 49200177A.
%  Fecha inicio: 20/04/2020
%  Fecha fin:

%% Descripción del programa

%% Parámetros del programa
f = 40;                        % Nº filas matriz spins, ha de ser par.
c = 40;                        % Nº columnas "     "  , ha de ser par.
n=f*c;

RNG = 1;                       % Generador de números aleatorios:
                               % 1 = Mersenne Twister, 
                               % 2 = Multiplicative Lagged Fibonacci.
                               
                               % Figuras:
figuras_variables=1;           % 1 = si, 0 = no.

opt_inicial=3;                 % Matriz de espines inicial: 
                               % 1 = Tablero de ajedrez.
                               % 2 = Aleatorio con probabilidad p_dw de apuntar abajo.
                               % 3 = Se toma como matriz inicial la matriz final de la temperatura anterior.
                                                           
opt_p_dw=0.98;                 % Si opt_p_dw pertenece a [0,1] entonces p_dw = opt_p_dw. Sinó se define un 
                               % p_dw(temperatura) de tal modo que la matriz de espines resulte en  
                               % una magnetización similar a la que indica la solución analítica.

%% Constantes físicas
kb=1;                          % Constante de Botlzmann en unidades en las que vale 1.
                               
%% Parámetros físicos
T_min=0.1;                       % Límite inferior del barrido de temperaturas de la simulación.
T_max=5;                       % Límite superior del ...
N_temps=100;                    % Número de puntos del ...
N_temps_a=500;                 % Número de puntos del barrido de temperaturas para la sol. analitica.

J = 1;                         % Constante de interacción entre spins en unidades en las que vale 1.
H = 0.01;                         % Campo magnético externo.

%% Parámetros numéricos
N_p = 1e5;                     % Nº pasos MMC máximos admitidos. Si el sistema no se ha termalizado se da por termalizado.
N_mean=2*n;
N_durante_eq=10*n;             % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.
seed=654618;                   % Semilla del RNG.                           
%% _______Programa_______
% Inicializar variables
Tc=(2/log(1+sqrt(2)));           % Temperatura crítica, se obliga a que forme parte del barrido de temperaturas.

                                 % Barrido de temperaturas de la simulacion MMC.
temps=[linspace(T_min,Tc,N_temps/2) linspace(Tc,T_max,N_temps/2)];
temps(length(temps)/2)=[];

                                 % Barrido de temperaturas de la sol. analítica.
temps_a=[linspace(T_min,Tc,N_temps_a) linspace(Tc,T_max,N_temps_a)];    
temps_a(length(temps_a)/2)=[];

                                 % Nº pasos a partir de los cuales se considera que el sistema esta termalizado.
                                 % Inicialmente vale N_p-N_durante_eq. Segun evoluciona el sistema esta variable se modifica.
                                 % Cuando el sistema esta termalizado se fija N_eq(T)=i.
Error=0.000001*temps;

N_eq = N_p*ones(size(temps))-N_durante_eq;    

N_p2 = N_p*ones(size(temps));

if opt_p_dw<0 || opt_p_dw>1    % Definición de la matriz inicial de espines en el caso que no sea tablero ajedrez.
p_dw=abs((1-(sinh(2./temps)).^(-4)).^(1/8));
pos=find(p_dw==0);
p_dw(pos:end)=0;
p_dw=0.495+p_dw*0.5;
else
p_dw=opt_p_dw*ones(size(temps));
end

if RNG==1
    rand_settings=rng(seed,'twister');
elseif RNG==2
    rand_settings=rng(seed,'multFibonacci');
end
E  = zeros(size(temps));
E2 = zeros(size(temps));
M  = zeros(size(temps));
M2 = zeros(size(temps));
Cv = zeros(size(temps));
X  = zeros(size(temps));

%% Simulación Metrópolis Monte Carlo para distintas temperaturas.
for T = 1:length(temps)
fprintf('Iteración %d de %d \n',T,length(temps))
% Inicializción de variables para cada temperatura.
eq=0;
Eeq  = [];
E2eq = [];
Meq  = [];
M2eq = [];
Eeq2  = [];
E2eq2 = [];
Meq2  = [];
M2eq2 = [];
k_m=0;

% Se crea la matriz de spins inicial.
if opt_inicial < 3 || T==1
s=config_inicial(f,c,opt_inicial,p_dw(T));
end
s2=s;
s2(1:end)=s2(randperm(n));

Ei = EnergiaIsing(s,J,H); %Energia inicial por espin.
Mi = mean(s(:));    %Magne inicial por espin.
E_t=Ei;
M_t=Mi;

Ei2 = EnergiaIsing(s2,J,H); %Energia inicial por espin.
Mi2 = mean(s2(:));    %Magne inicial por espin.
E_t2=Ei2;
M_t2=Mi2;

% Bucle pasos algoritmo Metropolis-MC
for i = 0:N_p
    
    % MMC Matriz 1
    f_p=ceil(rand*f);
    c_p=ceil(rand*c);
    dE=deltaEnergiaIsing(s,f_p,c_p,J,H);%Diferencia de energía total, no por spin.
    if dE<=0 || rand<exp(-dE/(kb*temps(T)))
        % Actualizo la matriz de espines, la energía y la magnetización.
        s(f_p,c_p)=-s(f_p,c_p);
        Ei=Ei+dE/(f*c);
        Mi=Mi+2*s(f_p,c_p)/(f*c);   
    end
    
    %MMC Matriz 2
    f_p2=ceil(rand*f);
    c_p2=ceil(rand*c);
    dE2=deltaEnergiaIsing(s2,f_p2,c_p2,J,H);%Diferencia de energía total, no por spin.
    if dE2<=0 || rand<exp(-dE2/(kb*temps(T)))
        % Actualizo la matriz de espines, la energía y la magnetización.
        s2(f_p2,c_p2)=-s2(f_p2,c_p2);
        Ei2=Ei2+dE2/n;
        Mi2=Mi2+2*s2(f_p2,c_p2)/n;   
    end
    
    E_t=[E_t Ei];
    M_t=[M_t Mi];
    
    E_t2=[E_t2 Ei2];
    M_t2=[M_t2 Mi2];
    
    if length(M_t)==N_mean+N_mean*k_m && eq==0
        k_m=k_m+1;
        C1=abs( abs(mean(M_t)) - abs(mean(M_t2)) );
        c1=C1 < Error(T);
        
        c2= abs(Mi)==1
       if c1 || c2
           fprintf('Equilibrio para T=%.2f alcanzado en %d iteraciones\n',temps(T),i)
           eq=1;
           N_eq(T)=i;
           N_p2(T)=i+N_durante_eq;
       end
    end
    
    if i > N_eq(T)
        Meq  = [Meq Mi];
        M2eq = [M2eq  Mi^2];
        Eeq  = [Eeq   Ei];
        E2eq = [E2eq  Ei^2];
        
        Meq2  = [Meq2 Mi2];
        M2eq2 = [M2eq2  Mi2^2];
        Eeq2  = [Eeq2   Ei2];
        E2eq2 = [E2eq2  Ei2^2];
    end
    
    if i==N_p2(T),break,end
       
end

M(T)=(mean(Meq)+mean(Meq2))/2;
M2(T)=(mean(M2eq)+mean(M2eq2))/2;
E(T)=(mean(Eeq)+mean(Eeq2))/2;
E2(T)=(mean(E2eq)+mean(E2eq2))/2;

end
Cv=(f*c)*(E2-E.^2)./(temps.^2);
X=(f*c)*(M2-M.^2)./(temps);

%% Solución analítica
T_a=linspace(0.1,(2/log(1+sqrt(2))),1000);
theta=linspace(0,pi/2,1000);
k_a=1./(sinh(2*T_a)).^2;

E_a=-coth(2./T_a).*( 1+(2/pi)*(2*(tanh(2./T_a)).^2-1)*...
    trapz(theta,1./sqrt(1-4*k_a.*(1+k_a).^(-2).*(sin(theta)).^2)) );
M_a=(1-(sinh(2./T_a)).^(-4)).^(1/8);

[F_anl, E_anl, dEdT]=Onsager(temps_a);

%% Figuras simulacion
if figuras_variables
figure
plot(temps,E,'.k')
hold on
plot(temps_a,E_anl,'-b','linewidth',1.5)
% plot(T_a,E_a,'-b','linewidth',1.5)
title('E')

figure
plot(temps,Cv,'.k')
hold on
plot(temps_a,dEdT/2,'-b','linewidth',1.5)
%plot(T_a,[0 diff(E_a)],'-b','linewidth',1.5)
title('Cv')
ylim([0, max(Cv(2:end))])

figure
plot(temps,-M,'.k')
hold on
plot(T_a,M_a,'-b','linewidth',1.5)
title('M')

figure
hold on
plot(temps,X,'.k')
% plot(temps,-[0 diff(M)],'.r')
% plot(T_a,-[0 diff(M_a)],'-b','linewidth',1.5)
title('X')
ylim([0, max(X)])
end






