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
modelo = 4;                     % 1 = Ising 2D con interacción a primeros vecinos, 
                                % 2 = SK 2D con interacción a primeros vecinos,
                                % 3 = SK 3D con interacción a primeros vecinos,
                                % 4 = SK n-D con interacción a primeros vecinos, es decir,
                                %     SK con interacción de rango infinito.
                                                                                                        
use_replicas = 1;
parallel_tempering = 1;
estudio_escalado = 0;

figura_variables=1;            % 1 = si, 0 = no.                 
figura_variables_vs_anal=0;           
figura_error=0;
figura_pasos=0;  

%% Parámetros físicos
f = 15;                        % Nº filas matriz de espines, ha de ser par.
c = f;                         % Nº columnas "     "  , ha de ser par. 
                               % Para el modelo 3, el 3D, hay restricciones en f y c:
                               % 1) f = c;
                               % 2) f^(1/6) tiene que ser exacto. ¿Por que? Porque 729 espines se pueden guardar como una
                               % red cubica de 9x9x9 o como una red cuadrada de 27x27.
                               % Usar f=8, f=27, f=64.

N_temps=20;                    % Número de puntos del barrido de temperaturas de la simulación.
T_min=0;                     % Límite inferior del barrido de temperaturas de la simulación.
T_max=2;                     % Límite superior del barrido de temperaturas de la simulación.


N_replicas=10; 

temps=linspace(T_min,T_max,N_temps); % Barrido de temperaturas de la simulacion MMC.
N_temps_a=1000;                % Número de puntos del barrido de temperaturas para la sol. analitica del modelo de Ising.

H = 0;                         % Campo magnético externo.
kb= 1;                         % Constante de Botlzmann en unidades en las que vale 1.
J = 1;
J0 = 0;
p_dw = 0.5;                    % La matriz de espines inicial es una matriz aleatoria 
                               % donde los espines tienen p_dw de apuntar hacia abajo.
                               
% L=ceil(logspace(1,2.3,20));    % Barrido de simulaciones para varias dimensiones [Lf x Lc].   
% L(rem(L,2)~=0)=L(rem(L,2)~=0)+1;
L = [ 8 12 16 20 24 ];
Lf= L;
Lc= L;

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
C_t = round(20*linspace(3,1,N_temps));%*ones(N_temps,1);         % Cuando hay C_t parejas de valores de E medidos a cada N_t pasos que difieren menos de 
Error_t=temps/sqrt(n);     % Error_t se considera que el sistema está termalizado.
N_durante_eq=1000*n;   % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.
N_pt = 100*n; % A cada N_pt se plantea un cambio de parallel tempering entre dos simulaciones a distinta T.
N_indep = 10*n; % A cada N_indep se considera que el sistema ha evolicionado lo suficiente como para tomar otra muestra del mismo y utilizarla
                 % en los promedios termodinamicos.

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
E  = zeros(1,N_temps);
E2 = zeros(1,N_temps);
M  = zeros(1,N_temps);
M2 = zeros(1,N_temps);
Cv = zeros(1,N_temps);
X  = zeros(1,N_temps);
S  = zeros(f,c,N_temps);
Sr  = zeros(f,c,N_temps,N_replicas);

%% Replicas
Jm=Jmatrix(f,c,modelo,seed,RNG,J,J0);
s0=config_inicial(f,c,p_dw,seed); % Matriz inicial de espines.
for replica = 1:1+(N_replicas-1)*use_replicas
fprintf('Replica %d de %d.\n',replica,1+(N_replicas-1)*use_replicas)
% Jm=Jmatrix(f,c,modelo,seed*replica,RNG,J,J0);
s=s0;
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

E_vs_t=Ei;

% Bucle pasos algoritmo Metropolis-MC
while i<N_p2(T)
    
    if parallel_tempering && T>1
            labSend(s,T-1,0);
    end
    
for i = last_i+1:last_i+N_pt
    
%         if accept_tempering && T>1 && labProbe(T-1,1)
% %             fprintf('Cambio recibido de T=%.3f a T=%.3f. \n',temps(T-1),temps(T))
%                     s = labReceive(T-1,1);
%                     Ei = Energia(s,H,Jm,modelo); %Energia inicial por espin.
%                     Mi = mean(s(:));     %Magne inicial por espin.
%         end

    
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
        E_vs_t = [E_vs_t Ei];
    end
    
    if i>N_eq(T) && rem(i,N_indep)==0
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
                if rand<p  %&& ( E_pt<Ei || accept_tempering )
%                     labSend(s,T+1,1);
                    
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
q_c = mean(mi_c.^2,[1 2]);
g_c = (1/2)*( 3 - mean(mi_c.^4,[1 2])/(mean(mi_c.^2,[1 2])^2) );
end

for T =1:N_temps
M(T)=M_c{T};
M2(T)=M2_c{T};
E(T)=E_c{T};
E2(T)=E2_c{T};
q(T) =q_c{T};
S(:,:,T) = s{T};
mP(T) = mean(P{T});
g(T) = g_c{T};
end

Cv=(f*c)*(E2-E.^2)./(temps.^2);
X=(f*c)*(M2-M.^2)./(temps);

% Store replicas
if use_replicas
Er(replica,:) = E;
Mr(replica,:) = M;
M2r(replica,:)= M2;
Cvr(replica,:) = Cv;
Xr(replica,:) = X;
qr(replica,:)= q;
Sr(:,:,:,replica)=S;
gr(replica,:)= g;
end

end

if use_replicas
    E = mean(Er);
    M = mean(Mr);
    Cv = mean(Cvr);
    X = mean(Xr);
    M2 = mean(M2r);
    q = mean(qr);
    g = mean(gr);
    for T=1:N_temps
        for i = 1:N_replicas
            for j = 1:N_replicas
                if j==i, qp(i,j,T)=NaN; continue, end
                si = Sr(:,:,T,i);
                sj = Sr(:,:,T,j);
                qp(i,j,T)=mean( si.*sj ,[1 2]);
            end
        end
    end
    
end

%% Figuras simulacion
subplot = @(m,n,p) subtightplot (m, n, p, [0.08 0.08], [0.05 0.05], [.1 0.05]); %Not important, just tighten plots.

if figura_variables && ~estudio_escalado
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
plot(E_vs_t{i})
end
legend

figure
plot(temps,abs(g),'ob','markersize',2)
ylabel('$g$','interpreter','latex')

if parallel_tempering
    figure
    semilogy(temps,mP,'o')
end

% if use_replicas
%     figureqp
%     hold on
%     for T =1:N_temps
%         histogram(abs(qp(:,:,T)))
%     end
% end

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

    if estudio_escalado
    Ed(ii,:) = E;
    Md(ii,:) = M;
    M2d(ii,:)= M2;
    Cvd(ii,:) = Cv;
    Xd(ii,:) = X;
    qd(ii,:)= q;
    gd(ii,:)= g;
    end

end

if estudio_escalado
    
figure
subplot(2,2,1)
plot(temps,Ed,'-.','markersize',2)
ylabel('$E$','interpreter','latex')
% ylim([min(Ed,[1 2]) max(Ed,[1 2])])
pbaspect([1 0.8 1])

subplot(2,2,2)
plot(temps,Cvd,'-.','markersize',2)
ylabel('$C_V$','interpreter','latex')
% ylim([0, max(Cv(2:end))])
pbaspect([1 0.8 1])

subplot(2,2,3)
plot(temps,abs(Md),'-.','markersize',2)
 ylabel('$M$','interpreter','latex')
pbaspect([1 0.8 1])

subplot(2,2,4)
plot(temps,Xd,'-.','markersize',2)
ylabel('$\chi$','interpreter','latex')
% ylim([0, max(Xd,[1 2])])
pbaspect([1 0.8 1])
drawnow
print('magnitudes10x10.png','-dpng','-r500')

figure
plot(temps,qd,'-.','markersize',10)
ylabel('$q$','interpreter','latex')

figure
plot(temps,abs(gd),'-.','markersize',10)
ylabel('$g$','interpreter','latex')    
    
end
toc
