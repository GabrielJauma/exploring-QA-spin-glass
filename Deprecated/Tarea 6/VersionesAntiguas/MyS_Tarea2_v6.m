clc, clear, close all, warning('off')
%% Máster en Física de sistemas complejos, UNED
%  Modelización y simulación de sistemas complejos
%  Tarea 2. Modelo de Ising en dos dimensiones

%  Autor: Gabriel Jaumà Gómez, 49200177A.
%  Fecha inicio: 20/04/2020
%  Fecha fin:

%% Descripción del programa

%% Parámetros del programa
RNG = 1;                       % Generador de números aleatorios:
                               % 1 = Mersenne Twister, 
                               % 2 = Multiplicative Lagged Fibonacci.
                               
figura_spins=0;                % Figuras:
figuras_variables=1;           % 1 = si, 0 = no.
figura_termalizacion=0;
N_f=1;                       % Se actualiza la figura a cada N_f cambios.

opt_inicial=3;                 % Matriz de espines inicial: 
                               % 1 = Tablero de ajedrez.
                               % 2 = Aleatorio con probabilidad p_dw de apuntar abajo.
                               % 3 = Se toma como matriz inicial la matriz final de la temperatura anterior.
                                                           
opt_p_dw=0.99;                   % Si opt_p_dw pertenece a [0,1] entonces p_dw = opt_p_dw. Sinó se define un 
                               % p_dw(temperatura) de tal modo que la matriz de espines resulte en  
                               % una magnetización similar a la que indica la solución analítica.

%% Constantes físicas
kb=1;                          % Constante de Botlzmann en unidades en las que vale 1.
                               
%% Parámetros físicos
T_min=0;                       % Límite inferior del barrido de temperaturas de la simulación.
T_max=5;                       % Límite superior del ...
N_temps=40;                    % Número de puntos del ...
N_temps_a=500;                 % Número de puntos del barrido de temperaturas para la sol. analitica.

J = 1;                         % Constante de interacción entre spins en unidades en las que vale 1.
H = 0;                         % Campo magnético externo.

%% Parámetros numéricos
f = 40;                        % Nº filas matriz spins, ha de ser par.
c = 40;                        % Nº columnas "     "  , ha de ser par.
N_p = 1e6;                     % Nº pasos MMC máximos admitidos. Si el sistema no se ha termalizado se da por termalizado.
N_mean=1e4;                    % Calcula la media de E y M a cada N_mean pasos y decide si esta termalizado o no.
N_durante_eq=1e3;              % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.

error_E=0.1;                   % Errores en la E y M para que se acepte como estado de equilibrio, luego son multiplicados por
error_M=0.01;                  % un factor que depende de la temperatura y hace que el error admitido sea 0 para T=0 y
                               % sea error_M para T=T_max.
seed=4;                        % Semilla del RNG.                           
%% _______Programa_______
% Inicializar variables

Tc=(2/log(1+sqrt(2)));           % Temperatura crítica, se obliga a que forme parte del barrido de temperaturas.

                                 % Barrido de temperaturas de la simulacion MMC.
temps=[linspace(T_min,Tc,N_temps/2) linspace(Tc,T_max,N_temps/2)];
temps(length(temps)/2)=[];

                                 % Barrido de temperaturas de la sol. analítica.
temps_a=[linspace(T_min,Tc,N_temps_a) linspace(Tc,T_max,N_temps_a)];    
temps_a(length(temps_a)/2)=[];

error_E=error_E*(1-exp(-temps))+0.001; % Error en la E para que se acepte como estado de equilibrio.
error_M=error_M*(1-exp(-temps))+0.001; % Error en la M ...

                                 % Nº pasos a partir de los cuales se considera que el sistema esta termalizado.
                                 % Inicialmente vale N_p-N_durante_eq. Segun evoluciona el sistema esta variable se modifica.
                                 % Cuando el sistema esta termalizado se fija N_eq(T)=i.
N_eq = N_p*ones(size(temps))-N_durante_eq;    

N_p2 = N_p*ones(size(temps));
Eprobabilidad=zeros(length(temps),N_durante_eq);

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
p  = zeros(size(temps));

%% Simulación Metrópolis Monte Carlo para distintas temperaturas.
for T = 1:length(temps)
fprintf('Iteración %d de %d \n',T,length(temps))
% Inicializción de variables para cada temperatura.
k_eq = 0;        % Contador de iteraciones "en equilibrio".
k_mc=0;          % Guardo la ultima iteración en la que se hizo un cambio.
k_f=0;           % Contador para las figuras.
eq=0;
E_t_mean1=-10;
M_t_mean1=-10;
i_MC=0;

% Se crea la matriz de spins inicial.
if opt_inicial < 3 || T==1
s=config_inicial(f,c,opt_inicial,p_dw(T));
end
Ei = EnergiaIsing(s,J,H); %Energia inicial por espin.
Mi = mean(s(:));    %Magne inicial por espin.

E_t=Ei;
M_t=Mi;

% Bucle pasos algoritmo Metropolis-MC
for i = 0:N_p
    f_p=ceil(rand*f);
    c_p=ceil(rand*c);
    dE=deltaEnergiaIsing(s,f_p,c_p,J,H);%Diferencia de energía total, no por spin.

    if dE<=0 || rand<exp(-dE/(kb*temps(T)))
        k_mc=i; 
        i_MC=i_MC+1;
        
        % Actualizo la matriz de espines, la energía y la magnetización.
        s(f_p,c_p)=-s(f_p,c_p);
        Ei=Ei+dE/(f*c);
        Mi=Mi+2*s(f_p,c_p)/(f*c);
        
        % Evolucion temporal de E y M con el algoritmo metropolis.
        k_f=k_f+1;
        if k_f>N_f && figura_termalizacion
            hold on
            plot(i,Ei,'k.')
            plot(i,Mi,'b.')
            drawnow
            k_f=0;
        end
        
        % Representacion blanco y negro de los spins.
        if k_f>N_f && figura_spins
            clf
            colormap([0 0 0; 1 1 1 ]);
            image(s .* 255);
            axis('equal');
            pbaspect([1 1 1]);
            title(sprintf('T=%f, i=%d, iMC=%d, i / iMC=%f',temps(T),i,i_MC,i/i_MC));
            drawnow
            k_f=0;
        end 
    end
    
    % En cada paso guardo la energía y la magnetización, independientemente
    % de si cambia o no, lo uso para analizar la evolucion.
    E_t=[E_t Ei];
    M_t=[M_t Mi];
    
    % A cada N_mean pasos se calcula la media de la energía y de la magnetización por spin.
    % Se compara la media de las segundas N_mean iteraciones con la media de
    % las a primeras N_mean iteraciones, si esta diferencia es menor que
    % error_mean se considera que el sistema esta termalizado. En caso
    % contrario se compara con las siguientes N_mean iteraciones y así
    % sucesivamente.
    
    if length(E_t)==N_mean && eq==0
        E_t_mean2=mean(E_t);
        M_t_mean2=mean(M_t);
%         fprintf('%.3f, %.3f\n',E_t_mean1,M_t_mean1);
%         fprintf('%.3f, %.3f\n',E_t_mean2,M_t_mean2);
%         disp('----------------');
        if abs(E_t_mean1-E_t_mean2)<error_E(T) && abs(M_t_mean1-M_t_mean2)<error_M(T)
            N_eq(T)=i;
            N_p2(T)=N_eq(T)+N_durante_eq;
            eq=1;
            fprintf('Equilibrio para T=%.2f alcanzado en %d pasos por equilibrio E y M \n',temps(T),i)
            E_t=[];
            M_t=[];
        else
            E_t_mean1=E_t_mean2;
            E_t=[];
            M_t_mean1=M_t_mean2;
            M_t=[];
         end
    end
    
    
    % Cuando han pasado N_eq pasos MC se considera que el sist. está
    % termalizado y se tienen en cuenta la E y la M para los cálculos.
    
    if i > N_eq(T)
        k_eq = k_eq + 1; %Contador de iteraciones "en equilibrio".
        M(T)  = M(T)  + abs(Mi);
        M2(T) = M2(T) + Mi^2;
        E(T)  = E(T)  + Ei;
        E2(T) = E2(T) + Ei^2;
        Eprobabilidad(T,k_eq)=Ei;
    end
    
    if i==N_p2(T),break,end
       
end

M(T)=abs(M(T))/k_eq;
M2(T)=M2(T)/k_eq;
E(T)=E(T)/k_eq;
E2(T)=E2(T)/k_eq;
p(T)=i/i_MC;

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

%% Verificacion de distribución de Boltzmann de probabilidad.
% figure
% hold on
% for T=1:length(temps)
%    P=[0 0];
%    EpT=Eprobabilidad(T,:);
%    for i=1:length(EpT)
%        if all(P(:,1)~=EpT(i))
%        P=[P
%           EpT(i) length(find(EpT==EpT(i)))];
%        end
%    end
%    P(1,:)=[];
%    Ee =P(:,1)*f*c;
%    %[Ee, I]=sort(P(:,1));
%    NEe=P(:,2);
%    %NEe=NEe(I);
%    if T==1 E0=Ee(1); end
%    plot(Ee-E0,NEe/(N_durante_eq),'k.')
%    plot(Ee-E0,exp((E0-Ee)/temps(T))/max(exp((E0-Ee)/temps(T))),'b-')
% end

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
plot(temps,M,'.k')
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

% figure
% semilogy(temps,N_eq,'.k')
% title('N_eq')
% 
% figure
% plot(temps,p/max(p),'.k')
% hold on
% plot(temps_a,exp(-temps_a),'linewidth',1.5)
% title('i / iMC')
end






