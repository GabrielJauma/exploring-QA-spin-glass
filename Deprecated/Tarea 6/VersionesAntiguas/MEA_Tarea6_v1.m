clc, clear, close all, warning('off')
%% Máster en Física de sistemas complejos, UNED
%  Modelización y simulación de sistemas complejos
%  Tarea 2. Modelo de Ising en dos dimensiones

%  Autor: Gabriel Jaumà Gómez, 49200177A.
%  Fecha inicio: 20/04/2020
%  Fecha fin: 03/05/2020

%% Descripción del programa
%  Este programa ha sido diseñado para simular el modelo de Ising en dos
%  dimensiones, haciendo uso del algoritmo de Metrópolis - Monte Carlo.

%% Parámetros del programa

modelo = 3;                     % 1 = Ising, 2 = Vidrio de espín Sherrington-Kirkpatrick (SK) con interacción a primeros vecinos,
                                % 3 = SK con interacción de rango infinito.

L=ceil(logspace(1,2.3,20));
L(rem(L,2)~=0)=L(rem(L,2)~=0)+1;
dim=[];                         % Barrido de simulaciones para varias dimensiones, el tamaño de la matriz de espines será 
                                % (dim)x(dim). Si está vacio la dimension se especifica a mano en f y c.
                               
f = 20;                        % Nº filas matriz spins, ha de ser par.
c = 20;                        % Nº columnas "     "  , ha de ser par. 

criterio_termalizacion=1;      % 1 = Se acepta el sistema midiendo la M a cada cierto número de pasos y estudiando su evolución.
                               % Vease mas abajo los parámetros que ajustan estos criterios.
                              
figura_variables=1;            % 1 = si, 0 = no.                 
figura_variables_vs_anal=0;           
figura_error=0;
figura_pasos=0;
figuras_temporales=0;

opt_inicial=2;                 % Matriz de espines inicial: 
                               % 1 = Tablero de ajedrez.
                               % 2 = Aleatorio con probabilidad p_dw de apuntar abajo.
                               % 3 = Se toma como matriz inicial la matriz final de la temperatura anterior.
                                                           
opt_p_dw=0.9;                  % Si opt_p_dw pertenece a [0,1] entonces p_dw = opt_p_dw. Sinó se define un 
                               % p_dw(temperatura) de tal modo que la matriz de espines resulte en  
                               % una magnetización igual a la que indica la solución analítica.
                                                             
%% Parámetros físicos
T_min=0.1;                     % Límite inferior del barrido de temperaturas de la simulación.
T_max=5;                       % Límite superior del barrido de temperaturas de la simulación.
H = 0;                         % Campo magnético externo.
kb= 1;                         % Constante de Botlzmann en unidades en las que vale 1.
J = 1;
J0 = 10;

%% Parámetros numéricos
N_temps=6;                    % Número de puntos del barrido de temperaturas de la simulación.
N_temps_a=1000;                % Número de puntos del barrido de temperaturas para la sol. analitica.
RNG = 1;                       % Generador de números aleatorios:
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
N_p = 1e8;                  % Nº pasos MMC máximos admitidos. Si el sistema no se ha termalizado se da por termalizado.
N_t = 10*n;                    % A cada N_t pasos se mide M. Las medidas se usan para decidir sobre la termalización.
C_t = 10;                    % Cuando hay C_t parejas de valores de M medidos a cada N_t pasos que difieren menos de 
Error_t=1e-3;               % Error_t se considera que el sistema está termalizado.
N_durante_eq=10*n;          % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% _______Programa_______
% Inicializar variables
J=Jmatrix(f,c,modelo,seed,RNG,J,J0);

Tc=(2/log(1+sqrt(2)));           % Temperatura crítica, se obliga a que forme parte del barrido de temperaturas.

if N_temps>1
temps=[linspace(T_min,Tc,N_temps/2) linspace(Tc,T_max,N_temps/2)]; % Barrido de temperaturas de la simulacion MMC.
temps(length(temps)/2)=[];                      
else
temps = 5; 
end

% Sol analitica
if N_temps>1 && modelo<3
temps_a=[linspace(T_min,Tc,N_temps_a) linspace(Tc,T_max,N_temps_a)];% Barrido de temperaturas de la sol. analítica.    
temps_a(length(temps_a)/2)=[];
[E_anal, ~ , ~]=Onsager(temps,1);
end

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
parfor T = 1:length(temps)
% fprintf('Iteración %d de %d \n',T,length(temps))
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

Ei = Energia(s,H,J,modelo); %Energia inicial por espin.
Mi = mean(s(:));     %Magne inicial por espin.

if figuras_temporales
    figure(1)
    clf
end

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
    dE=deltaEnergia(s,f_p,c_p,H,J,modelo);%Diferencia de energía total, no por spin.
    
    if dE<=0 || rand<exp(-dE/(kb*temps(T)))
        % Actualizo la matriz de espines, la energía y la magnetización.
        s(f_p,c_p)=-s(f_p,c_p);
        Ei=Ei+dE/(f*c);
        Mi=Mi+2*s(f_p,c_p)/(f*c);   
    end
    
    if figuras_temporales && rem(i,10000)==0
        drawnow
        
        figure(1)
        hold on
        subplot(2,1,1)
        plot(i,Ei,'.r')
        hold on
        subplot(2,1,2)
        plot(i,Mi,'.b') 
    end
    
    if criterio_termalizacion == 1
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
subplot = @(m,n,p) subtightplot (m, n, p, [0.08 0.08], [0.05 0.05], [.1 0.05]); %Not important, just tighten plots.

if figura_variables
% Figuras variables
figure
subplot(2,2,1)
plot(temps,E,'ok','markersize',2)
ylabel('$E$','interpreter','latex')
ylim([min(E) max(E)])
pbaspect([1 0.8 1])

subplot(2,2,2)
plot(temps,Cv,'ok','markersize',2)
ylabel('$C_V$','interpreter','latex')
ylim([0, max(Cv(2:end))])
pbaspect([1 0.8 1])

subplot(2,2,3)
plot(temps,abs(M),'ok','markersize',2)
ylabel('$M$','interpreter','latex')
pbaspect([1 0.8 1])

subplot(2,2,4)
plot(temps,X,'ok','markersize',2)
ylabel('$\chi$','interpreter','latex')
ylim([0, max(X)])
pbaspect([1 0.8 1])
drawnow
print('magnitudes10x10.png','-dpng','-r500')

figure
semilogy(temps,N_eq,'.k')
end

if figura_variables_vs_anal
% Solución analítica
[E_a, Cv_a, M_a]=Onsager(temps_a,1);    

% Figuras variables
figure
subplot(2,2,1)
hold on
plot([Tc Tc],[min(E) max(E)],'-r','linewidth',1)
plot(temps_a,E_a,'-b','linewidth',1.5)
plot(temps,E,'ok','markersize',2)
ylabel('$E$','interpreter','latex')
ylim([min(E) max(E)])
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
drawnow
print('magnitudes10x10.png','-dpng','-r500')

figure
semilogy(temps,N_eq,'.k')
end

if figura_error && length(dim)>1
% Solución analítica
[E_a, Cv_a, M_a]=Onsager(temps,1); 
% Error
Error_E=abs(E-E_a);
Error_E_x(ii)=max(Error_E);
Error_E_m(ii)=mean(Error_E);

Error_M=abs(abs(M)-abs(M_a));
Error_M_x(ii)=max(Error_M);
Error_M_m(ii)=mean(Error_M);

Error_Cv=abs(Cv-Cv_a);
Error_Cv_x(ii)=max(Error_Cv);
Error_Cv_m(ii)=mean(Error_Cv);

%Error_Tc(ii)=temps(find(Cv==max(Cv)))-Tc;
Error_Tc(ii)=temps(Cv==max(Cv))-Tc;
end

if figura_pasos && length(dim)>1
Pasos_m(ii)= mean(N_eq);
Pasos_x(ii)= max(N_eq);
end

end

if figura_error && length(dim)>1 
figure
subplot(2,2,1)
[f_fit,formula]=log_fit(Error_E_m,dim.^2,4,length(dim.^2)); 
loglog(dim.^2, f_fit,'b')
hold on
loglog(dim.^2,Error_E_m,'-k')
legend({formula},'interpreter','latex','location','northeast')
ylabel('$\langle|E(T)-E_a(T)|\rangle$','interpreter','latex')

subplot(2,2,2)
[f_fit,formula]=log_fit(Error_M_m,dim.^2,4,length(dim.^2)); 
loglog(dim.^2, f_fit,'b')
hold on
loglog(dim.^2,Error_M_m,'-k')
ylabel('$\langle|M(T)-M_a(T)|\rangle$','interpreter','latex')
legend({formula},'interpreter','latex','location','northeast')

subplot(2,2,3)
semilogx(dim.^2,Error_Cv_m,'-k')
ylabel('$\langle|Cv(T)-Cv_a(T)|\rangle$','interpreter','latex')

subplot(2,2,4)
semilogx(dim.^2,abs(Error_Tc),'-k')
ylabel('$\langle|T(T)-T_c(T)|\rangle$','interpreter','latex')

print('Errorvsdim.png','-dpng','-r500')
end

if figura_pasos && length(dim)>1
   figure
   [f_fit,formula]=log_fit(Pasos_m,dim.^2,4,length(dim.^2)); 
   loglog(dim.^2, f_fit,'b')
   hold on
   loglog(dim.^2,Pasos_m,'ok','markersize',4) 
   ylabel('$N_{eq}$','interpreter','latex')
   legend({formula},'interpreter','latex','location','northwest')
   ylim([min(Pasos_m) max(Pasos_m)])
   xlim([min(dim.^2) max(dim.^2)])
   set(gca,'FontSize',16)
   print('NumeroPasos.png','-dpng','-r500')
end



