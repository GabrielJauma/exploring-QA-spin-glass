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

use_conf=1;                               
use_replicas = 1;
parallel_tempering = 1;
estudio_escalado = 1;

figura_variables=1;            % 1 = si, 0 = no.                 
figura_variables_vs_anal=0;   
figure_distributions=0;
figura_error=0;
figura_pasos=0;  

%% Parámetros físicos
f = 6;                        % Nº filas matriz de espines, ha de ser par.
c = f;                         % Nº columnas "     "  , ha de ser par. 
                               % Para el modelo 3, el 3D, hay restricciones en f y c:
                               % 1) f = c;
                               % 2) f^(1/6) tiene que ser exacto. ¿Por que? Porque 729 espines se pueden guardar como una
                               % red cubica de 9x9x9 o como una red cuadrada de 27x27.
                               % Usar f=8, f=27, f=64.

N_temps=20;                    % Número de puntos del barrido de temperaturas de la simulación.
T_min=0.6;                     % Límite inferior del barrido de temperaturas de la simulación.
T_max=1.4;                     % Límite superior del barrido de temperaturas de la simulación.


N_replicas=2; 
N_conf=2;

temps=linspace(T_min,T_max,N_temps); % Barrido de temperaturas de la simulacion MMC.
N_temps_a=1000;                % Número de puntos del barrido de temperaturas para la sol. analitica del modelo de Ising.

H = 0;                         % Campo magnético externo.
kb= 1;                         % Constante de Botlzmann en unidades en las que vale 1.
J = 1;
J0 = 0;
p_dw = 0.5;                    % La matriz de espines inicial es una matriz aleatoria 
                               % donde los espines tienen p_dw de apuntar hacia abajo.                             
%L = [ 8 10 12 14 16 18 20 ];   % Barrido de simulaciones para varias dimensiones [Lf x Lc]. 
L =[6 10];
Lf= L;
Lc= L;
%% Parámetros numéricos
RNG = 1;                       % Generador de números aleatorios:
                               % 1 = Mersenne Twister, 2 = Multiplicative Lagged Fibonacci.
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

%% PARAMETROS TERMALIZACIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_p = 1e8;       % Nº pasos MMC máximos admitidos. Si el sistema no se ha termalizado se da por termalizado.
N_t = round(10*n);       % A cada N_t pasos se mide E. Las medidas se usan para decidir sobre la termalización.
C_t = round(50*linspace(3,1,N_temps));%*ones(N_temps,1);         % Cuando hay C_t parejas de valores de E medidos a cada N_t pasos que difieren menos de 
Error_t=sqrt(temps/n);     % Error_t se considera que el sistema está termalizado.
N_durante_eq=round(10000*n);   % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.
N_pt = round(100*n); % A cada N_pt se plantea un cambio de parallel tempering entre dos simulaciones a distinta T.
N_indep =n; %round(sqrt(n)); % A cada N_indep se considera que el sistema ha evolicionado lo suficiente como para tomar otra muestra del mismo y utilizarla
N_indep =N_indep  + rem(N_durante_eq,N_indep);                 % en los promedios termodinamicos.

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
S  = zeros(n,N_durante_eq/N_indep,N_replicas,N_temps);
Sc = zeros(n,N_durante_eq/N_indep,N_replicas);
%% Configurations ( of J )

for conf = 1:1+(N_conf-1)*use_conf
Jm=Jmatrix(n,modelo,seed*conf,RNG,J,J0);
s0=config_inicial(f,c,p_dw,seed*conf); % Matriz inicial de espines.

%% Replicas
for replica = 1:1+(N_replicas-1)*use_replicas
clc
fprintf('Red %dx%d.\n',dim(ii),dim(ii))
fprintf('Replica %d de %d.\n',replica,1+(N_replicas-1)*use_replicas)
fprintf('Configuracion %d de %d.\n',conf,1+(N_conf-1)*use_conf)
s=s0;
Ei = Energia(s,H,Jm,modelo); %Energia inicial por espin.
Mi = mean(s(:));     %Magne inicial por espin.

%% Temperatures
% Simulación Metrópolis - Monte Carlo para distintas temperaturas.
spmd
T=labindex;
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
send_tempering=1;
E_vs_t=Ei;
proposed_swaps=0;
accepted_swaps=0;

% Bucle pasos algoritmo Metropolis-MC
while i<N_p2(T)
    
    if T>1 && labProbe(T-1,1) 
        labReceive(T-1,1);
        send_tempering=0;
    end
    
    if parallel_tempering && T>1  && send_tempering && accept_tempering
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
%         E_vs_t = [E_vs_t Ei];
    end
    
    if i>N_eq(T) && rem(i,N_indep)==0 && k_eq<(N_durante_eq/N_indep)
        k_eq = k_eq + 1;
        Meq  = Meq  + Mi;
        M2eq = M2eq + Mi^2;
        Eeq  = Eeq  + Ei;
        E2eq = E2eq + Ei^2;
        
        mi = mi+s;
        Sc(:,k_eq,replica) =s(:);
        E_vs_t = [E_vs_t Ei];
    end
 
        k_t=k_t+1;
        if k_t==N_t && eq==0
            k_t=0;
            c_t = c_t + double(abs(abs(Ei)-abs(Ei2))<Error_t(T));
                if abs(abs(Ei)-abs(Ei2))<Error_t(T) && rem(c_t,5)==0
%                     fprintf('T=%.2f, %.2f %% completado\n',temps(T),c_t*100/C_t(T))
                end
            Ei2=Ei;
            if c_t==C_t(T)
                N_eq(T)=i;
                N_pt = N_durante_eq;
                N_p2(T)=i+N_durante_eq;
                eq=1;
                if parallel_tempering && T<N_temps
                        accept_tempering=0;
                        labSend(accept_tempering,T+1,1);
                end
            end
        end
    
%     if i==N_p2(T),fprintf('T=%.2f, equilibrio en %.2e iteraciones. \n',temps(T),i),break,end

end
last_i=i;
%     clc

if parallel_tempering && T<N_temps && accept_tempering 
        if labProbe(T+1,0)          
            s_pt = labReceive(T+1,0);
            E_pt = Energia(s_pt,H,Jm,modelo);
            p = min([1, exp( (Ei-E_pt)*( 1/(kb*temps(T))- 1/(kb*temps(T+1))  )  )]);
            proposed_swaps=proposed_swaps+1;
%             fprintf('Cambio propuesto de T=%.3f a T=%.3f\n con %2.e probabilidad de aceptacion.\n',temps(T+1),temps(T),p)
                if rand<p  && or(temps(T)>0, E_pt<Ei)
                    accepted_swaps=accepted_swaps+1;
                    s = s_pt;
                    Ei = Energia(s,H,Jm,modelo); %Energia inicial por espin.
                    Mi = mean(s(:));     %Magne inicial por espin.
%                     fprintf('Cambio aceptado de T=%.3f a T=%.3f con %.3f probabilidad de aceptación. \n',temps(T+1),temps(T),p)
                end
        end       
        
end

end

%% Purgue messages
if T<N_temps
    while labProbe(T+1,0)
        labReceive(T+1,0);
    end 
end
if T>1
    while labProbe(T-1,1)
        labReceive(T-1,1);
    end 
end




M_T=Meq/k_eq;
M2_T=M2eq/k_eq;
E_T=Eeq/k_eq;
E2_T=E2eq/k_eq;
mi_T = mi/k_eq;
q_T = mean(mi_T.^2,[1 2]);
end


for T =1:N_temps
M(T)=M_T{T};
M2(T)=M2_T{T};
E(T)=E_T{T};
E2(T)=E2_T{T};
q(T) =q_T{T};
S(:,:,:,T) = Sc{T};

mP(T) = accepted_swaps{T}/proposed_swaps{T};

end

Cv=(f*c)*(E2-E.^2)./(temps.^2);
X=(f*c)*(M2-M.^2)./(temps);

% Store replicas
if use_replicas
Er(replica,:) = E;
Mr(replica,:) = M;
Cvr(replica,:) = Cv;
Xr(replica,:) = X;
qr(replica,:)= q;
end

end

if use_replicas
    E = mean(Er);
    M = mean(Mr);
    Cv = mean(Cvr);
    X = mean(Xr);
    q = mean(qr);
    for T=1:N_temps
        for t=1:(N_durante_eq/N_indep)
            for a = 1:N_replicas
             sa = S(:,t,a,T);
                for b = 1:N_replicas
                    sb    = S(:,t,b,T);
                    q1(a,b,t)=mean(sa.*sb);
                    q2(a,b,t) = mean(sa.*sb)^2;
                    q4(a,b,t) =q2(a,b,t)^2;
                end
            end
        end
        %Thermal average
        q1_t_av = mean(q1,3);
        q2_t_av = mean(q2,3);
        q4_t_av = mean(q4,3);
        q4q2_t_av= q4_t_av./(q2_t_av.^2);
        %Replica average
        q1_r_av(T)=mean(q1_t_av, [1 2])*N_replicas^2/( N_replicas^2 -N_replicas)-N_replicas/( N_replicas^2 -N_replicas);
        q2_r_av(T) =  mean(q2_t_av, [1 2])*N_replicas^2/( N_replicas^2 -N_replicas)-N_replicas/( N_replicas^2 -N_replicas);
        q4_r_av(T) =  mean(q4_t_av, [1 2])*N_replicas^2/( N_replicas^2 -N_replicas)-N_replicas/( N_replicas^2 -N_replicas);
        q4q2_r_av(T) = mean(q4q2_t_av, [1 2])*N_replicas^2/( N_replicas^2 -N_replicas)-N_replicas/( N_replicas^2 -N_replicas);
        
        g(T) =  0.5*(3 - q4q2_r_av(T));
        B(T) =  0.5*(3 - q4_r_av(T)/q2_r_av(T)^2);
    end
      
end

    if use_conf
    Ec(conf,:) = E;
    Mc(conf,:) = M;
    Cvc(conf,:) = Cv;
    Xc(conf,:) = X;
    qc(conf,:)= q;    
    
    q2_c(conf,:)=q2_r_av;
    q4_c(conf,:)=q4_r_av;
    end
end

if use_conf
    q2_c_av=mean(q2_c,1);
    q4_c_av=mean(q4_c,1);
    q4q2_c_av=mean(q4_c_av./(q2_c_av.^2),1);
    
    Bc = 0.5*(3 - q4_c_av./(q2_c_av.^2) );
    gc = 0.5*(3 - q4q2_c_av);

    E = mean(Ec);
    M = mean(Mc);
    Cv= mean(Cvc);
    X = mean(Xc);
    q = mean(qc); 
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
hold on
ylabel('$q$','interpreter','latex')

figure
hold on
for i=1:4
plot(E_vs_t{i})
end
legend

if use_replicas
    figure
    Bf=fit(temps',B','gauss1');
    plot(temps,B,'ob','markersize',2)
    hold on
    plot(Bf)
    ylabel('$B$','interpreter','latex')
end

if use_conf
    figure
    Bcf=fit(temps',Bc','gauss1');
    plot(temps,Bc,'ob','markersize',2)
    hold on
    plot(Bcf)
    ylabel('$Bc$','interpreter','latex')

    figure
    gcf=fit(temps',gc','gauss1');
    plot(temps,gc,'ob','markersize',2)
    hold on
    plot(gcf)
    ylabel('$gc$','interpreter','latex')
    
    figure
    plot(temps,Qc,'ob','markersize',2)
    ylabel('$Qc$','interpreter','latex')
end

if parallel_tempering
    figure
    semilogy(temps,mP)
    ylabel('Acceptance ratio','interpreter','latex')
    xlabel('$T$','interpreter','latex')
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

    if estudio_escalado
    Ed(ii,:) = E;
    Md(ii,:) = M;
    M2d(ii,:)= M2;
    Cvd(ii,:)= Cv;
    Xd(ii,:) = X;
    qd(ii,:) = q;
    Bd(ii,:) = Bc;
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
plot(temps,abs(Md),'-','markersize',2)
ylim([0 1])
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


colors=hot( size(dim,1)+3 );
figure
hold on
for ii=1:size(dim,1)
Bf=fit(temps',Bd(ii,:)','gauss1');
plot(temps,Bd(ii,:),'o','markersize',5,'color',colors(ii,:))
fig_fit=plot(Bf)
set(fig_fit,'color',colors(ii,:),'linewidth',1.5)
ylabel('$B$','interpreter','latex')  
end
    
end
toc

save('Resultados.dat')

if figure_distributions
    colors=jet( N_temps+2);
    fig=figure;
    hold on
    x_dist=linspace(min(E_vs_t{1}),max(E_vs_t{20}),1000);
    for i=1:N_temps
    dist=fitdist(E_vs_t{i}','normal');
    variances(i)=var(dist);
    plot(x_dist, pdf(dist,x_dist),'color',colors(i,:),'linewidth',1.5)   
    end 
    xlabel('$E(T)$','interpreter','latex')
    ylabel('$p(E)$','interpreter','latex')
    set(gca,'fontsize',14)
    print('Gausianas.png','-dpng','-r500')
    
    figure
    plot(temps,sqrt(variances),'.')
    hold on
    plot(temps,Error_t)
end
