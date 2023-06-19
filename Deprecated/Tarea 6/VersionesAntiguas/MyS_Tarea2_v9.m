clc, clear, close all, warning('off')
%% Máster en Física de sistemas complejos, UNED
%  Modelización y simulación de sistemas complejos
%  Tarea 2. Modelo de Ising en dos dimensiones

%  Autor: Gabriel Jaumà Gómez, 49200177A.
%  Fecha inicio: 20/04/2020
%  Fecha fin:

%% Descripción del programa

%% Parámetros del programa
dim=[ ];                       % Barrido de simulaciones para varias dimensiones, el tamaño de la matriz de espines será 
                               % (dim)x(dim). Si está vacio la dimension se especifica a mano en f y c.
f = 10;                        % Nº filas matriz spins, ha de ser par.
c = 10;                        % Nº columnas "     "  , ha de ser par. 

criterio_termalizacion=2;      % 1 = Se acepta el sistema cuando el error en la E frente a la sol analítica es menor que un cierto valor.
                               % 2 = Se acepta el sistema midiendo la M a cada cierto número de pasos y estudiando su evolución.
                               % Vease mas abajo los parámetros que ajustan estos criterios.
                               
figuras_variables=1;           % 1 = si, 0 = no.
figuras_error=0;

opt_inicial=3;                 % Matriz de espines inicial: 
                               % 1 = Tablero de ajedrez.
                               % 2 = Aleatorio con probabilidad p_dw de apuntar abajo.
                               % 3 = Se toma como matriz inicial la matriz final de la temperatura anterior.
                                                           
opt_p_dw=0.9;                  % Si opt_p_dw pertenece a [0,1] entonces p_dw = opt_p_dw. Sinó se define un 
                               % p_dw(temperatura) de tal modo que la matriz de espines resulte en  
                               % una magnetización igual a la que indica la solución analítica.
                                                             
%% Parámetros físicos
T_min=0.1;                     % Límite inferior del barrido de temperaturas de la simulación.
T_max=5;                       % Límite superior del barrido de temperaturas de la simulación.
J = 1;                         % Constante de interacción entre spins en unidades en las que vale 1.
H = 0;                         % Campo magnético externo.
kb=1;                          % Constante de Botlzmann en unidades en las que vale 1.

%% Parámetros numéricos
N_temps=100;                   % Número de puntos del barrido de temperaturas de la simulación.
N_temps_a=1000;                % Número de puntos del barrido de temperaturas para la sol. analitica.
RNG = 2;                       % Generador de números aleatorios:
                               % 1 = Mersenne Twister, 
                               % 2 = Multiplicative Lagged Fibonacci.
seed=654618;                   % Semilla del RNG. 

if isempty(dim)
    dim=f;
end

for ii=1:length(dim)
f=dim(ii);
c=dim(ii);
n=f*c;                         % Nº de elementos ...

fprintf('\n Red %dx%d.\n',dim(ii),dim(ii))
%% PARAMETROS TERMALIZACIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_p = 1e7;                     % Nº pasos MMC máximos admitidos. Si el sistema no se ha termalizado se da por termalizado.
N_t = 2*n;                     % A cada N_t pasos se mide M. Las medidas se usan para decidir sobre la termalización.
C_t = 10;                      % Cuando hay C_t parejas de valores de M medidos a cada N_t pasos que difieren menos de 
Error_t=1e-3;                  % Error_t se considera que el sistema está termalizado.
N_durante_eq=1000*n;           % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% _______Programa_______
% Inicializar variables
Tc=(2*J/log(1+sqrt(2)));           % Temperatura crítica.                            
temps=linspace(T_min,T_max,N_temps);  % Barrido de temperaturas de la simulacion MMC.                       
temps_a=[linspace(T_min,Tc,N_temps_a) linspace(Tc,T_max,N_temps_a)];% Barrido de temperaturas de la sol. analítica.    
temps_a(length(temps_a)/2)=[];

[E_anal, ~ , ~]=Onsager(temps,J);

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
%fprintf('Iteración %d de %d \n',T,length(temps))
% Inicializción de variables para cada temperatura.
eq = 0;
Eeq  = 0;
E2eq = 0;
Meq  = 0;
M2eq = 0;
k_eq = 0;
k_t  = 0;
Mi2 = 100;
E_t=[];
c_t=0;

% Se crea la matriz de spins inicial.
if opt_inicial < 3 || T==1
s=config_inicial(f,c,opt_inicial,p_dw(T));
end

Ei = EnergiaIsing(s,J,H); %Energia inicial por espin.
Mi = mean(s(:));    %Magne inicial por espin.

% Bucle pasos algoritmo Metropolis-MC
for i = 0:N_p
    if i > N_eq(T)
        k_eq = k_eq + 1;
        Meq  = Meq  + Mi;
        M2eq = M2eq + Mi^2;
        Eeq  = Eeq  + Ei;
        E2eq = E2eq + Ei^2;
    end
    
    % MMC
    f_p=ceil(rand*f);
    c_p=ceil(rand*c);
    dE=deltaEnergiaIsing(s,f_p,c_p,J,H);%Diferencia de energía total, no por spin.
    if dE<=0 || rand<exp(-dE/(kb*temps(T)))
        % Actualizo la matriz de espines, la energía y la magnetización.
        s(f_p,c_p)=-s(f_p,c_p);
        Ei=Ei+dE/(f*c);
        Mi=Mi+2*s(f_p,c_p)/(f*c);   
    end
    
    if criterio_termalizacion == 2
        k_t=k_t+1;
        if k_t==N_t && eq==0
            k_t=0;
            c_t = c_t + int8(abs(abs(Mi)-abs(Mi2))<Error_t);
            Mi2=Mi;
            if c_t==C_t
                N_eq(T)=i;
                N_p2(T)=i+N_durante_eq;
                eq=1;
                fprintf('T=%.2f, eq=%d \n',temps(T),i);
            end
        end
    elseif criterio_termalizacion==1
        E_t=[E_t Ei];
        if length(E_t)==N_t
            if abs(mean(E_t)-E_anal(T))<Error_t
                N_eq(T)=i;
                N_p2(T)=i+N_durante_eq;
                eq=1;
                fprintf('T=%.2f, eq=%d \n',temps(T),i);    
            else 
            E_t=[];
            end
        end
    end
    
    if i==N_p2(T),break,end
    
end

M(T)=Meq/k_eq;
M2(T)=M2eq/k_eq;
E(T)=Eeq/k_eq;
E2(T)=E2eq/k_eq;

end

Cv=(f*c)*(E2-E.^2)./(temps.^2);
X=(f*c)*(M2-M.^2)./(temps);


%% Figuras simulacion
subplot = @(m,n,p) subtightplot (m, n, p, [0.08 0.08], [0.05 0.05], [.1 0.05]); %Not important, jus tighten plots.

if figuras_variables
% Solución analítica
[E_a, Cv_a, M_a]=Onsager(temps_a,J);    

% Figuras variables
subplot(2,2,1)
hold on
plot([Tc Tc],[min(E) max(E)],'-r','linewidth',1)
plot(temps_a,E_a,'-b','linewidth',1.5)
plot(temps,E,'ok','markersize',2)
ylabel('$E$','interpreter','latex')
pbaspect([1 0.8 1])

subplot(2,2,2)
hold on
plot([Tc Tc],[min(Cv) max(Cv)],'-r','linewidth',1)
plot(temps_a,Cv_a,'-b','linewidth',1.5)
plot(temps,Cv,'ok','markersize',2)
ylabel('$C_V$','interpreter','latex')
ylim([0, max(Cv(2:end))])
pbaspect([1 0.8 1])

subplot(2,2,3)
hold on
plot([Tc Tc],[min(abs(M)) max(abs(M))],'-r','linewidth',1)
plot(temps_a,M_a,'-b','linewidth',1.5)
plot(temps,abs(M),'ok','markersize',2)
ylabel('$M$','interpreter','latex')
pbaspect([1 0.8 1])

subplot(2,2,4)
hold on
plot([Tc Tc],[min(X) max(X)],'-r','linewidth',1)
plot(temps,X,'ok','markersize',2)
ylabel('$\chi$','interpreter','latex')
ylim([0, max(X)])
pbaspect([1 0.8 1])

print('magnitudes200x200.png','-dpng','-r500')
end

if figuras_error
% Solución analítica
[E_a, Cv_a, M_a]=Onsager(temps,J); 
% Error
Error_E=abs(E-E_a);
Error_E_x(ii)=max(Error_E);
Error_E_m(ii)=mean(Error_E);
end

end

if figuras_error
figure
plot(dim,Error_E_x,'-r')
hold on
plot(dim,Error_E_m,'-k')
end


