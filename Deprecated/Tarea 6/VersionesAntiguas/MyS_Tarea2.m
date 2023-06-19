clc, clear, close all
%% Máster en Física de sistemas complejos, UNED
%  Modelización y simulación de sistemas complejos
%  Tarea 2. Modelo de Ising en dos dimensiones

%  Autor: Gabriel Jaumà Gómez, 49200177A.
%  Fecha inicio: 20/04/2020
%  Fecha fin:

%% Descripción del programa

%% Parámetros del programa
RNG = 1;                       % 1 = Mersenne Twister, 
                               % 2 = Multiplicative Lagged Fibonacci.
                               
figura_spins=0;                % Figuras.
figuras_variables=1;           % 1 = si, 0 = no.
figura_termalizacion=1;

opt_inicial=1;                 % Matriz de spins inicial. 
p_dw=0.5;                      % 1 = Tablero de ajedrez.
                               % 2 = Aleatorio con probabilidad
                               % p_dw de apuntar abajo.
%% Parámetros numéricos
f = 40;                       % Nº filas matriz spins, ha de ser par.
c = 40;                       % Nº columnas "     "  , ha de ser par.
N_p = 1e5;                    % Nº pasos Monte Carlo totales, los primeros
                              % N_eq pasos son para termalizar. Para los
                              % resultados de energía, magnetiz, ..., solo
                              % cuentan los (N_p-N_eq) pasos restantes.
seed=4;                       % Semilla del RNG.
N_mean=1e3;                   % Calcula la media a cada N_mean pasos y decide si esta termalizado o no.
N_durante_eq=1e3;             % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.

%% Parámetros físicos
temps=linspace(0,5,20);     % Barrido de temperaturas de la simulacion.
J = 1;                       % Constante de interacción entre spins.
H = -0.10;                % Campo magnético externo.

%% Constantes físicas
%kb = 1.38064852e-23;        %
kb=1;


%% Detalles de Matlab
N_eq = N_p*ones(size(temps))-N_durante_eq;        % Nº pasos a partir de los cuales se considera que el sistema esta termalizado
N_p2 = N_p*ones(size(temps));
k_f=0;
figure
hold on

%% _______Programa_______
% Inicializar variables
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

% Simulación MC para distintas temperaturas.
for T = 1:length(temps)
fprintf('Iteración %d de %d \n',T,length(temps))
% Inicializción de variables para cada temperatura.
beta = 1.0/(kb*temps(T));
k_eq = 0;
eq=0;
E_t=[];
M_t=[];
E_t_mean1=-10;
M_t_mean1=-10;

% Se crea la matriz de spins inicial.
s=config_inicial(f,c,opt_inicial,p_dw);

% Bucle pasos algoritmo Metropolis-MC
tic
for i = 0:N_p
   % Cuando han pasado N_eq pasos MC se considera que el sist. está
   % termalizado y se tienen en cuenta la E y la M para los cálculos.
   
   if i==N_p2(T),break,end
   
   if i > N_eq(T) 
       k_eq = k_eq + 1; %Contador de iteraciones "en equilibrio".
       M(T)  = M(T)  + mean(s(:));
       M2(T) = M2(T) + mean(s(:))^2;
       E(T)  = E(T)  + EnergiaIsing(s,J,H);
       E2(T) = E2(T) + EnergiaIsing(s,J,H)^2;   
   end
   
   % Algoritmo Metropolis-MC
   f_p=ceil(rand*f);
   c_p=ceil(rand*c);
   sp=s;
   sp(f_p,c_p)=-sp(f_p,c_p);
   Ei = EnergiaIsing(s,J,H);
   Ep = EnergiaIsing(sp,J,H);
   dE=(Ep-Ei)*(f*c); %Diferencia de energía total, no por spin.
   
   if dE<=0 || rand<exp(-dE/(kb*temps(T)))
   s=sp;
   
   E_t=[E_t Ep];
   M_t=[M_t mean(s(:))];
   % A cada N_mean cambios de configuracion se calcula la media de la energía por spin.
   % Se compara la media de las segundas N_mean iteraciones con la media de
   % las a primeras N_mean iteraciones, si esta diferencia es menor que
   % error_mean se considera que el sistema esta termalizado. En caso
   % contrario se compara con las siguientes N_mean iteraciones y así
   % sucesivamente.
   
   if length(E_t)==N_mean && eq==0
	   E_t_mean2=mean(E_t);
       M_t_mean2=mean(M_t);
       if abs(E_t_mean1-E_t_mean2)<0.1 && abs(M_t_mean1-M_t_mean2)<0.01
       N_eq(T)=i;
       N_p2(T)=N_eq(T)+N_durante_eq;
       eq=1;
       fprintf('Equilibrio para T=%.2f alcanzado en %d pasos \n',temps(T),i)
       E_t=[];
       M_t=[];
       else
       E_t_mean1=E_t_mean2;
       E_t=[];
       M_t_mean1=M_t_mean2;
       M_t=[];
       end  
   end
   
       % Evolucion temporal de E con el algoritmo metropolis.
       k_f=k_f+1;
       if k_f>100 && figura_termalizacion
       plot(i,mean(E_t),'k.')
       drawnow
       k_f=0;
       end
       
       % Representacion blanco y negro de los spins.
       if k_f>1000 && figura_spins
       clf
       colormap([0 0 0; 1 1 1 ]);
       image(s .* 255);
       axis('equal');
       pbaspect([1 1 1]);
       title(num2str(i));
       drawnow
       k_f=0;
       end
       
   end
end

time(T)=toc;

M(T)=M(T)/k_eq;
M2(T)=M2(T)/k_eq;
E(T)=E(T)/k_eq;
E2(T)=E2(T)/k_eq;

Cv(T)=E2(T)-E(T)^2;
X(T)=(M2(T)-M(T)^2)/beta;
end


if figuras_variables
figure
plot(temps,E,'.k')
title('E')

figure
plot(temps,Cv,'.k')
title('Cv')

figure
plot(temps,M,'.k')
title('M')

figure
plot(temps,X,'.k')
title('X')
end