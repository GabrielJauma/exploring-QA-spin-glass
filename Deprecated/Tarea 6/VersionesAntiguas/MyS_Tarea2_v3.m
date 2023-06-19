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
figura_termalizacion=0;

opt_inicial=2;                 % Matriz de spins inicial. 
                               % 1 = Tablero de ajedrez.
                               % 2 = Aleatorio con probabilidad
                               % p_dw de apuntar abajo.
                               
%% Parámetros físicos
N_temps=10;
Tc=(2/log(1+sqrt(2)));
temps=[linspace(0,Tc,N_temps/2) linspace(Tc,5,N_temps/2)];    % Barrido de temperaturas de la simulacion.
temps(length(temps)/2)=[];

temps_a=[linspace(0,Tc,500) linspace(Tc,5,500)];    % Barrido de temperaturas de la sol. analítica.
temps_a(length(temps_a)/2)=[];

J = 1;                       % Constante de interacción entre spins.
H = 0;                       % Campo magnético externo.

%% Parámetros numéricos
f = 40;                      % Nº filas matriz spins, ha de ser par.
c = 40;                      % Nº columnas "     "  , ha de ser par.
N_p = 1e7;                    % Nº pasos Monte Carlo totales, los primeros
                              % N_eq pasos son para termalizar. Para los
                              % resultados de energía, magnetiz, ..., solo
                              % cuentan los (N_p-N_eq) pasos restantes.
seed=4;                       % Semilla del RNG.
N_mean=2e3;                   % Calcula la media a cada N_mean pasos y decide si esta termalizado o no.
N_durante_eq=1e4;             % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.
error_E=0.1*(1-exp(-temps));
error_M=0.01*(1-exp(-temps));
p_dw=abs((1-(sinh(2./temps)).^(-4)).^(1/8));
pos=find(p_dw==0);
p_dw(pos:end)=0;
p_dw=0.45+p_dw*0.5;

%% Constantes físicas
%kb = 1.38064852e-23;        %
kb=1;

%% Detalles de Matlab
N_eq = N_p*ones(size(temps))-N_durante_eq;        % Nº pasos a partir de los cuales se considera que el sistema esta termalizado
N_p2 = N_p*ones(size(temps));
k_f=0;
Eprobabilidad=zeros(length(temps),N_durante_eq);

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
p  = zeros(size(temps));
time = zeros(size(temps));

% Simulación MC para distintas temperaturas.
for T = 1:length(temps)
fprintf('Iteración %d de %d \n',T,length(temps))
% Inicializción de variables para cada temperatura.
beta = 1.0/(kb*temps(T));
k_eq = 0;
k_mc=0;
eq=0;
E_t=[];
M_t=[];
E_t_mean1=-10;
M_t_mean1=-10;
i_MC=0;
i_correlation=0;

% Se crea la matriz de spins inicial.
s=config_inicial(f,c,opt_inicial,p_dw(T));

% Bucle pasos algoritmo Metropolis-MC
tic
for i = 0:N_p
    
    % Algoritmo Metropolis-MC
    f_p=ceil(rand*f);
    c_p=ceil(rand*c);
    
    if i ==0
    Ei = EnergiaIsing(s,J,H);
    end
    
    sp=s;
    sp(f_p,c_p)=-sp(f_p,c_p);
    Ep = EnergiaIsing(sp,J,H);
    dE=(Ep-Ei)*(f*c); %Diferencia de energía total, no por spin.
    dE2=deltaEnergiaIsing(s,f_p,c_p,J,H);
    if dE2-dE>0.00000001
        'Cagada'
    end
     
    if dE<=0 || rand<exp(-dE/(kb*temps(T)))
         i_correlation=i_correlation+1;
         if i_correlation==10
         i_correlation=0;
        k_mc=i;  %Guardo la ultima iteración en la que se hizo un cambio.
        i_MC=i_MC+1;
        
        s=sp;
        Ei=Ep;
        
        E_t=[E_t Ep];
        M_t=[M_t abs(mean(s(:)))];
        
        % A cada N_mean cambios de configuracion se calcula la media de la energía y de la magnetización por spin.
        % Se compara la media de las segundas N_mean iteraciones con la media de
        % las a primeras N_mean iteraciones, si esta diferencia es menor que
        % error_mean se considera que el sistema esta termalizado. En caso
        % contrario se compara con las siguientes N_mean iteraciones y así
        % sucesivamente.
        if length(E_t)==N_mean && eq==0
            E_t_mean2=mean(E_t);
            M_t_mean2=mean(M_t);
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
        
        % Evolucion temporal de E con el algoritmo metropolis.
        k_f=k_f+1;
        if k_f>1 && figura_termalizacion
            hold on
            plot(i,Ep,'k.')
            plot(i,mean(sp(:)),'b.')
            drawnow
            k_f=0;
        end
        
        % Representacion blanco y negro de los spins.
        if k_f>100 && figura_spins
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
    end
    
    if abs(Ep+2)<1e-2 && (abs(mean(s(:)))-1)<1e-2 && eq==0
        N_eq(T)=i;
        N_p2(T)=N_eq(T)+N_durante_eq;
        eq=1;
        fprintf('Equilibrio para T=%.2f alcanzado en %d pasos por E=-2, M=+/-1 \n',temps(T),i)
    end
    
    
    % Cuando han pasado N_eq pasos MC se considera que el sist. está
    % termalizado y se tienen en cuenta la E y la M para los cálculos.
    
    if i > N_eq(T)
        Energia=EnergiaIsing(s,J,H);
        k_eq = k_eq + 1; %Contador de iteraciones "en equilibrio".
        M(T)  = M(T)  + abs(mean(s(:)));
        M2(T) = M2(T) + mean(s(:))^2;
        E(T)  = E(T)  + Energia;
        E2(T) = E2(T) + Energia^2;
        Eprobabilidad(T,k_eq)=Energia;
    end
    
    if i==N_p2(T),break,end
    
end

time(T)=toc;

M(T)=M(T)/k_eq;
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
figure
hold on
for T=1:length(temps)
   P=[0 0];
   EpT=Eprobabilidad(T,:);
   for i=1:length(EpT)
       if all(P(:,1)~=EpT(i))
       P=[P
          EpT(i) length(find(EpT==EpT(i)))];
       end
   end
   P(1,:)=[];
   [Ee, I]=sort(P(:,1));
   NEe=P(:,2);
   NEe=NEe(I);
   plot(Ee,NEe/N_durante_eq,'k.')
end



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

figure
plot(temps,M,'.k')
hold on
plot(T_a,M_a,'-b','linewidth',1.5)
title('M')

figure
hold on
plot(temps,X,'.k')
plot(T_a,-[0 diff(M_a)],'-b','linewidth',1.5)
title('X')

figure
semilogy(temps,N_eq,'.k')
title('N_eq')

figure
plot(temps,p/max(p),'.k')
hold on
plot(temps_a,exp(-temps_a),'linewidth',1.5)
title('i / iMC')
end






