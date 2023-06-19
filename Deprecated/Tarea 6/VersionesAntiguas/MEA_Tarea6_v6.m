clc, clear, close all, warning('off')
fclose('all');
tic
%% Máster en Física de sistemas complejos, UNED
%  Mecánica Estadística Avanzada
%  Tarea 6. Simulación de un vidrio de espín

%  Autor: Gabriel Jaumà Gómez, 49200177A.
%  Fecha inicio: 11/01/2021
%  Fecha fin: 

%% Descripción del programa
%  Este programa ha sido diseñado para simular el modelo de vidrio de espín
%  de Sherrington y Kirkpatrick (SK) haciendo uso del algoritmo de 
%  Metrópolis - Monte Carlo y de la ténica parallel-tempering.

%% Parámetros del programa
modelo = 3;                     % 1 = Ising 2D con interacción a primeros vecinos, 
                                % 2 = SK 2D con interacción a primeros vecinos,
                                % 3 = SK con interacción de rango infinito.
                                                                                                        
use_replicas = 0;
parallel_tempering = 1;
estudio_escalado = 0;

figura_variables=1;            % 1 = si, 0 = no.                 
figura_variables_vs_anal=0;           
figura_error=0;
figura_pasos=0;  

%% Parámetros físicos
f = 20;                        % Nº filas matriz de espines, ha de ser par.
c = f;                         % Nº columnas "     "  , ha de ser par. 

N_temps=20;                    % Número de puntos del barrido de temperaturas de la simulación.
T_min=0.001;                     % Límite inferior del barrido de temperaturas de la simulación.
T_max=1;                       % Límite superior del barrido de temperaturas de la simulación.


N_replicas=1; 

temps=linspace(T_min,T_max,N_temps); % Barrido de temperaturas de la simulacion MMC.
N_temps_a=1000;                % Número de puntos del barrido de temperaturas para la sol. analitica del modelo de Ising.

H = 0;                         % Campo magnético externo.
kb= 1;                         % Constante de Botlzmann en unidades en las que vale 1.
J = 1;
J0 = 0;
p_dw = 0.5;                    % La matriz de espines inicial es una matriz aleatoria 
                               % donde los espines tienen p_dw de apuntar hacia abajo.
                               
L=ceil(logspace(1,2.3,20));    % Barrido de simulaciones para varias dimensiones [Lf x Lc].   
L(rem(L,2)~=0)=L(rem(L,2)~=0)+1;
Lf=L;
Lc=L;

%% Parámetros numéricos
RNG = 1;                       % Generador de números aleatorios:
                               % 1 = Mersenne Twister, 
                               % 2 = Multiplicative Lagged Fibonacci.
seed=6546158;                  % Semilla del RNG. 


%% Programa %%
if estudio_escalado
    dim=[Lf' Lc'];
else
    dim=[f c];
end

for ii=1:size(dim,1) % Bucle para hacer un estudio sobre el tamaño del sistema
f=dim(ii,1);
c=dim(ii,2);
n=f*c;                         % Nº de elementos ...

fprintf('\n Red %dx%d.\n',dim(ii),dim(ii))
%% PARAMETROS TERMALIZACIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_p = 1e8;       % Nº pasos MMC máximos admitidos. Si el sistema no se ha termalizado se da por termalizado.
N_t = 100*n;       % A cada N_t pasos se mide E. Las medidas se usan para decidir sobre la termalización.
C_t = round(10*linspace(2,1,N_temps).^2);%*ones(N_temps,1);         % Cuando hay C_t parejas de valores de E medidos a cada N_t pasos que difieren menos de 
Error_t=temps/sqrt(n);     % Error_t se considera que el sistema está termalizado.
N_durante_eq=1000*n;   % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.
N_pt = 10*n; % A cada N_pt se plantea un cambio de parallel tempering entre dos simulaciones a distinta T.

if T_min<0.1, Error_t(1)=Error_t(2); end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% _______Programa_______
% Inicializar variables          
N_eq = N_p*ones(size(temps))-N_durante_eq;    
N_p2 = N_p*ones(size(temps));

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

%% Replicas
Jm=Jmatrix(f,c,modelo,seed,RNG,J,J0);
for replica = 1:1+(N_replicas-1)*use_replicas
fprintf('Replica %d de %d.\n',replica,1+(N_replicas-1)*use_replicas)
% Jm=Jmatrix(f,c,modelo,seed*replica,RNG,J,J0);
s=config_inicial(f,c,p_dw,seed*replica); % Matriz inicial de espines.
Ei = Energia(s,H,Jm,modelo); %Energia inicial por espin.
Mi = mean(s(:));     %Magne inicial por espin.

%% Simulación Metrópolis - Monte Carlo para distintas temperaturas.
spmd
T=labindex;
% T=1;
% Inicializción de variables para cada temperatura.
eq = 0;
Eeq  = 0;
E2eq = 0;
Meq  = 0;
M2eq = 0;
k_eq = 0;
k_t  = 0;
Ei2 = 100;
c_t=0;


last_i=0;
i=0;

P=[];
mi = s;
accept_tempering=1;
% s_corr=s;
% corr=[];

Ee=Ei;

% Bucle pasos algoritmo Metropolis-MC
while i<N_p2(T)
    
    if parallel_tempering && T>1
            labSend(s,T-1,0);
    end
    
for i = last_i+1:last_i+N_pt
    
    % MMC
    f_p=ceil(rand*f);
    c_p=ceil(rand*c);
    dE=deltaEnergia(s,f_p,c_p,H,Jm,modelo);%Diferencia de energía total, no por spin.

    if dE<=0 || rand<exp(-dE/(kb*temps(T)))
        % Actualizo la matriz de espines, la energía y la magnetización.
        s(f_p,c_p)=-s(f_p,c_p);
        Ei=Ei+dE/n;
        Mi=Mi+2*s(f_p,c_p)/n;
    end
    
    if rem(i,10*n)==0
        Ee = [Ee Ei];
    end
    
    if i>N_eq(T) && rem(i,10*n)==0
        accept_tempering=0;
        k_eq = k_eq + 1;
        Meq  = Meq  + Mi;
        M2eq = M2eq + Mi^2;
        Eeq  = Eeq  + Ei;
        E2eq = E2eq + Ei^2;
        
        mi = mi+s;
    end
 
        k_t=k_t+1;
        if k_t==N_t && eq==0
            k_t=0;
            c_t = c_t + double(abs(abs(Ei)-abs(Ei2))<Error_t(T));
                if abs(abs(Ei)-abs(Ei2))<Error_t(T)
                    fprintf('T=%.2f, %.2f %% completado\n',temps(T),c_t*100/C_t(T))
                end
            Ei2=Ei;
            if c_t==C_t(T)
                N_eq(T)=i;
                N_pt = N_durante_eq;
                N_p2(T)=i+N_durante_eq;
                eq=1;
            end
        end
    
    if i==N_p2(T),fprintf('T=%.2f, equilibrio en %.2e iteraciones. \n',temps(T),i),break,end

end
last_i=i;
%     clc

if parallel_tempering && T<N_temps && accept_tempering

        if labProbe(T+1,0)          
            s_pt = labReceive(T+1,0);
            E_pt = Energia(s_pt,H,Jm,modelo);
            p = min([1, exp( (Ei-E_pt)*( 1/(kb*temps(T))- 1/(kb*temps(T+1))  )  )]);
%             fprintf('Cambio propuesto de T=%.3f a T=%.3f\n con %2.e probabilidad de aceptacion.\n',temps(T+1),temps(T),p)
                if rand<p
                    P = [P p];
                    s = s_pt;
                    Ei = Energia(s,H,Jm,modelo); %Energia inicial por espin.
                    Mi = mean(s(:));     %Magne inicial por espin.
%                     fprintf('Cambio aceptado de T=%.3f a T=%.3f con %.3f probabilidad de aceptación. \n',temps(T+1),temps(T),p)
                end
        end

end

end

M_c=Meq/k_eq;
M2_c=M2eq/k_eq;
E_c=Eeq/k_eq;
E2_c=E2eq/k_eq;
mi_c = mi/k_eq;
q_c = mean(mean(mi_c.^2));
end

for T =1:N_temps
M(T)=M_c{T};
M2(T)=M2_c{T};
E(T)=E_c{T};
E2(T)=E2_c{T};
q(T) =q_c{T};
S(:,:,T) = s{T};
end

Cv=(f*c)*(E2-E.^2)./(temps.^2);
X=(f*c)*(M2-M.^2)./(temps);

% Store replicas
Er(replica,:) = E;
Mr(replica,:) = M;
M2r(replica,:)= M2;
Cvr(replica,:) = Cv;
Xr(replica,:) = X;
qr(replica,:)=q;
Sr(:,:,:,replica)=S;

end

if use_replicas
    E = mean(Er);
    M = mean(Mr);
    Cv = mean(Cvr);
    X = mean(Xr);
    M2 = mean(M2r);
    q = mean(qr);
    
    for T=1:N_temps
        for i = 1:N_replicas
            for j = 1:N_replicas
                if j==i, qp(i,j,T)=NaN, continue, end
                si = Sr(:,:,T,i);
                sj = Sr(:,:,T,j);
                qp(i,j,T)=mean( si.*sj ,[1 2]);
            end
        end
    end
end

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
plot(temps,q,'ok','markersize',2)
ylabel('$q$','interpreter','latex')

figure
hold on
for i=1:4
plot(Ee{i})
end
legend

if parallel_tempering
    figure
    hold on
    for i=1:N_temps
%        histogram(P{i})
    plot(temps(i),mean(P{i}),'o')
    end
end

if use_replicas
    figure
    hold on
    for T =1:N_temps
        histogram(qp(:,:,T),3)
    end
end

end

if figura_variables_vs_anal
% Solucion analitica al modelo de Ising 2D
Tc=(2/log(1+sqrt(2)));   
temps_a=[linspace(T_min,Tc,N_temps_a) linspace(Tc,T_max,N_temps_a)];% Barrido de temperaturas de la sol. analítica.    
temps_a(length(temps_a)/2)=[];
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
toc
