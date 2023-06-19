clc, clear, warning('off')
fclose('all');
tic
%% Máster en Física de sistemas complejos, UNED
%  Mecánica Estadística Avanzada
%  Tarea 6. Simulación de un vidrio de espín

%  Autor: Gabriel Jaumà Gómez, 49 200177A.
%  Fecha inicio: 11/01/2021
%  Fecha fin:  01/02/2021

%% Descripción del programa
%  Este programa ha sido diseñado para simular varios modelos de vidrios de espín,
%  incluido el de Sherrington y Kirkpatrick (SK), haciendo uso del algoritmo de 
%  Metrópolis - Monte Carlo y de la ténica parallel-tempering.

%% Parámetros del programa
modelo = 3;                     % 1 = Ising 2D con interacción a primeros vecinos, 
                                % 2 = vidrio de espin 2D con interacción a primeros vecinos,
                                % 3 = vidrio de espin 3D con interacción a primeros vecinos,
                                % 4 = vidrio de espin nD con interacción a primeros vecinos, 
                                %     es decir, modelo SK.
                                
estudio_escalado = 1;
use_conf=1;                               
use_replicas = 1;
parallel_tempering = 0;


figura_variables=1;            % 1 = si, 0 = no.                 
figura_variables_vs_anal=0;   
figure_distributions=0;

%% Parámetros físicos
f = 3;                         % Nº filas matriz de espines, ha de ser par.
c = f;                         % Nº columnas "     "  , ha de ser par. 
                               % Para el modelo 3, el 3D, f es el lado del cubo.

N_temps=8;                    % Número de puntos del barrido de temperaturas de la simulación.
T_min=0.8;                       % Límite inferior del barrido de temperaturas de la simulación.
T_max=1.1;                       % Límite superior del barrido de temperaturas de la simulación.
temps=linspace(T_min,T_max,N_temps);

N_replicas=2; 
N_conf=1000;

H = 0;                         % Campo magnético externo.
kb= 1;                         % Constante de Botlzmann en unidades en las que vale 1.
J = 1;
J0 = 0;
p_dw = 0.5;                    % La matriz de espines inicial es una matriz aleatoria 
                               % donde los espines tienen p_dw de apuntar hacia abajo.                             

L =[3 4];                % Barrido de simulaciones para varias dimensiones [Lf x Lc]. 
%L =[3 5]% 5 6]; %modelo 3
Lf= L;                         % ¡OJO! Si modelo==3 -> n=L^3.
Lc= L;
%% Parámetros numéricos
RNG = 1;                       % Generador de números aleatorios:
                               % 1 = Mersenne Twister, 2 = Multiplicative Lagged Fibonacci.
seed=9241368;                   % Semilla del RNG. 

%% Programa %%
if estudio_escalado
    dim=[Lf' Lc'];
else
    dim=[f c];
end

%% -BUCLE- Tamaño del sistema
for ii=1:size(dim,1) 
f=dim(ii,1);
c=dim(ii,2);
n=f*c; 

% chapucilla 
if modelo == 3, n=f^3; end
% end chapucilla

%% PARAMETROS TERMALIZACIÓN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_p = 1e8;                              % Nº pasos MMC máximos admitidos. Si el sistema no se ha termalizado se da por termalizado.
N_t = 100*n;                            % A cada N_t pasos se mide E. Las medidas se usan para decidir sobre la termalización.
C_t = round(10*linspace(3,1,N_temps));  % Cuando hay C_t parejas de valores de E medidos a cada N_t pasos que difieren menos de 
Error_t=sqrt(temps/n);                  % Error_t se considera que el sistema está termalizado.
N_durante_eq=1e3*n;                     % Numero de pasos utilizados para promediar las variables una vez el sistema esta termalizado.
N_pt0 = 10*n;                            % A cada N_pt se plantea un cambio de parallel tempering entre dos simulaciones a distinta T.
N_indep =n;                             % A cada N_indep se considera que el sistema ha evolicionado lo suficiente como para tomar otra muestra del mismo
N_indep =N_indep  + rem(N_durante_eq,N_indep);  % y utilizarla en los promedios termodinamicos.

if T_min<0.1, Error_t(1)=Error_t(2); end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Inicializar variables          
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

%% -BUCLE- Configuraciones ( de Jij )
for conf = 1:1+(N_conf-1)*use_conf
Jm=Jmatrix(n,modelo,seed*conf,RNG,J,J0);
s0=config_inicial(f,c,p_dw,seed*conf); % Matriz inicial de espines.
% chapucilla 
if modelo == 3, s0=config_inicial(n,1,p_dw,seed*conf); end
% end chapucilla

%% -BUCLE- Replicas
for replica = 1:1+(N_replicas-1)*use_replicas
clc
fprintf('Modelo %d .\n',modelo)
fprintf('%d espines.\n',n)
fprintf('Tamaño %d de %d.\n',ii,size(dim,1))
fprintf('Configuracion %d de %d.\n',conf,1+(N_conf-1)*use_conf)
fprintf('Replica %d de %d.\n',replica,1+(N_replicas-1)*use_replicas)
s=s0;
Ei = Energia(s,H,Jm,modelo); %Energia inicial por espin.
Mi = mean(s(:));     %Magne inicial por espin.

%% -PARALELO- Temperaturas
% Simulación Metrópolis - Monte Carlo para distintas temperaturas con
% intercambios Parallel Tempering.
spmd
T=labindex;

N_pt=N_pt0;
N_eq = N_p-N_durante_eq;    
N_p2 = N_p;
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

while i<N_p2 % While exterior que permite hacer parallel tempering
    
    %%  Check if it has to stop sending parallel tempering proposals
    if T>1 && labProbe(T-1,1) 
        labReceive(T-1,1);
        send_tempering=0;
    end
  
    %% Send Parallel Tempering proposal 
    if parallel_tempering && T>1  && send_tempering && accept_tempering
            labSend(s,T-1,0);
    end

    %% -BUCLE- Se plantea un cambio parallel tempering al final de cada ejecución completa del bucle
    for i = last_i+1:last_i+N_pt   
        %% MMC

        if modelo ~= 3
            f_p=ceil(rand*f);
            c_p=ceil(rand*c);
        else  % chapucilla para que (f*(c_p-1) + f_p) = f_p
            c_p=1; 
            f_p=ceil(rand*n); 
        end
        % end chapucilla
        
        dE=deltaEnergia(s,f_p,c_p,H,Jm,modelo);%Diferencia de energía total, no por spin.

        if dE<=0 || rand<exp(-dE/(kb*temps(T)))
            % Actualizo la matriz de espines, la energía y la magnetización.
            % (f*(c_p-1) + f_p) es el indice correspondiente a (f_p,c_p)
            s(f*(c_p-1) + f_p)=-s(f*(c_p-1) + f_p);
            Ei=Ei+dE/n;
            Mi=Mi+2*s(f*(c_p-1) + f_p)/n;
        end

        %% Store varaibles for thermal averages
        if i>N_eq && rem(i,N_indep)==0 && k_eq<(N_durante_eq/N_indep)
            k_eq = k_eq + 1;
            Meq  = Meq  + Mi;
            M2eq = M2eq + Mi^2;
            Eeq  = Eeq  + Ei;
            E2eq = E2eq + Ei^2;
            mi = mi+s;
            Sc(:,k_eq,replica) =s(:);
            E_vs_t = [E_vs_t Ei];
        end

        %% Decide about thermal equilibrium
            k_t=k_t+1;
            if k_t==N_t && eq==0
                k_t=0;
                c_t = c_t + double(abs(abs(Ei)-abs(Ei2))<Error_t(T));
    %                 if abs(abs(Ei)-abs(Ei2))<Error_t(T) && rem(c_t,C_t(T)/4)==0
    %                     fprintf('T=%.2f, %.2f %% completado\n',temps(T),c_t*100/C_t(T))
    %                 end
                Ei2=Ei;
                if c_t==C_t(T)
                    N_eq=i;
                    N_pt = N_durante_eq;
                    N_p2=i+N_durante_eq;
                    eq=1;
                    if parallel_tempering && T<N_temps
                            accept_tempering=0;
                            labSend(accept_tempering,T+1,1);
                    end
                end
            end
    end
    last_i=i;

    %% Recive Parallel Tempering proposal
    if parallel_tempering && T<N_temps && accept_tempering 
            if labProbe(T+1,0) 
                s_pt = labReceive(T+1,0);
                E_pt = Energia(s_pt,H,Jm,modelo);
                p = min([1, exp( (Ei-E_pt)*( 1/(kb*temps(T))- 1/(kb*temps(T+1))  )  )]);
                proposed_swaps=proposed_swaps+1;
                    if rand<p  %&& or(temps(T)>0, E_pt<Ei)
                        accepted_swaps=accepted_swaps+1;
                        s = s_pt;
                        Ei = Energia(s,H,Jm,modelo);
                        Mi = mean(s(:));
                    end
            end        
    end
end

%% Borrar mensajes perdidos de los workers de la paralelizacion
if parallel_tempering
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
end

%% Promedios termodinámicos
M_T  = Meq/k_eq;
M2_T = M2eq/k_eq;
E_T  = Eeq/k_eq;
E2_T = E2eq/k_eq;
mi_T = mi/k_eq;
q_T  = mean(mi_T.^2,[1 2]);
end


for T =1:N_temps
    M(T)  = M_T{T};
    M2(T) = M2_T{T};
    E(T)  = E_T{T};
    E2(T) = E2_T{T};
    q(T)  = q_T{T};
    S(:,:,:,T) = Sc{T};
    mP(T) = accepted_swaps{T}/proposed_swaps{T};
end
Cv = (f*c)*(E2-E.^2)./(temps.^2);
X  = (f*c)*(M2-M.^2)./(temps);

if use_replicas
    Er(replica,:)  = E;
    Mr(replica,:)  = M;
    Cvr(replica,:) = Cv;
    Xr(replica,:)  = X;
    qr(replica,:)  = q;
end

end

if use_replicas
    E  = mean(Er);
    M  = mean(Mr);
    Cv = mean(Cvr);
    X  = mean(Xr);
    q  = mean(qr);
    for T=1:N_temps
        for t=1:(N_durante_eq/N_indep)
            for a = 1:N_replicas
             sa = S(:,t,a,T);
                for b = 1:N_replicas
                    sb    = S(:,t,b,T);
                    q1(a,b,t) = mean(sa.*sb);
                    q2(a,b,t) = mean(sa.*sb)^2;
                    q4(a,b,t) = q2(a,b,t)^2;
                end
            end
        end
        %Thermal average
        q1_t_av   = mean(q1,3);
        q2_t_av   = mean(q2,3);
        q4_t_av   = mean(q4,3);
        q4q2_t_av = q4_t_av./(q2_t_av.^2);
        %Replica average
        q1_r_av(T) = mean(q1_t_av, [1 2])*N_replicas^2/( N_replicas^2 -N_replicas)-N_replicas/( N_replicas^2 -N_replicas);
        q2_r_av(T) = mean(q2_t_av, [1 2])*N_replicas^2/( N_replicas^2 -N_replicas)-N_replicas/( N_replicas^2 -N_replicas);
        q4_r_av(T) = mean(q4_t_av, [1 2])*N_replicas^2/( N_replicas^2 -N_replicas)-N_replicas/( N_replicas^2 -N_replicas);
        q4q2_r_av(T) = mean(q4q2_t_av, [1 2])*N_replicas^2/( N_replicas^2 -N_replicas)-N_replicas/( N_replicas^2 -N_replicas);
        
        B(T) =  0.5*(3 - q4_r_av(T)/q2_r_av(T)^2);
    end
      
end

if use_conf
    Ec(conf,:)  = E;
    Mc(conf,:)  = M;
    Cvc(conf,:) = Cv;
    Xc(conf,:)  = X;
    qc(conf,:)  = q;
    if use_replicas
        q1_c(conf,:)= q1_r_av;
        q2_c(conf,:)= q2_r_av;
        q4_c(conf,:)= q4_r_av;
        q4q2_c(conf,:)=q4q2_r_av;
    end
     B_c(conf,:) = 0.5*(3 - q4_r_av./(q2_r_av.^2) );
     g_c(conf,:) = 0.5*(3-
end

end

if use_conf
    E = mean(Ec);
    M = mean(Mc);
    Cv= mean(Cvc);
    X = mean(Xc);
    q = mean(qc); 
    if use_replicas
        q2_c_av=mean(q2_c,1);
        q4_c_av=mean(q4_c,1);
        q4q2_c_av=mean(q4_c_av./(q2_c_av.^2),1);

        Bc = 0.5*(3 - q4_c_av./(q2_c_av.^2) );
        gc = 0.5*(3 - q4q2_c_av);
    end
end

%% Figuras simulacion
subplot = @(m,n,p) subtightplot (m, n, p, [0.08 0.08], [0.05 0.05], [.1 0.05]); %Not important, just tighten plots.

if figura_variables && ~estudio_escalado
% Figuras variables
figure
subplot(2,2,1)
plot(temps,E,'-ok','markersize',3,'linewidth',2,'Color',[0.5,0.5,0.5],'MarkerEdgeColor','k','MarkerFaceColor','k')
ylabel('$E$','interpreter','latex')
ylim([min(E) max(E)])
pbaspect([1 0.8 1])

subplot(2,2,2)
plot(temps,Cv,'-ok','markersize',3,'linewidth',2,'Color',[0.5,0.5,0.5],'MarkerEdgeColor','k','MarkerFaceColor','k')
ylabel('$C_V$','interpreter','latex')
ylim([0, max(Cv(2:end))])
pbaspect([1 0.8 1])

subplot(2,2,3)
plot(temps,M,'-ok','markersize',3,'linewidth',2,'Color',[0.5,0.5,0.5],'MarkerEdgeColor','k','MarkerFaceColor','k')
ylabel('$M$','interpreter','latex')
pbaspect([1 0.8 1])

subplot(2,2,4)
plot(temps,X,'-ok','markersize',3,'linewidth',2,'Color',[0.5,0.5,0.5],'MarkerEdgeColor','k','MarkerFaceColor','k')
ylabel('$\chi$','interpreter','latex')
ylim([0, max(X)])
pbaspect([1 0.8 1])
drawnow
print('Figuras\Magnitudes_SK_n64_T0a2.png','-dpng','-r500')

figure
semilogy(temps,q,'-ok','markersize',3,'linewidth',2,'Color',[0.5,0.5,0.5],'MarkerEdgeColor','k','MarkerFaceColor','k')
hold on
ylabel('$q_{EA}$','interpreter','latex')
set(gca,'fontsize',14)
print('Figuras\qEA_SK_n64_T0a2_log.png','-dpng','-r500')

figure
plot(temps,q,'-ok','markersize',3,'linewidth',2,'Color',[0.5,0.5,0.5],'MarkerEdgeColor','k','MarkerFaceColor','k')
hold on
ylabel('$q_{EA}$','interpreter','latex')
ylim([0 1])
set(gca,'fontsize',14)
print('Figuras\qEA_SK_n64_T0a2.png','-dpng','-r500')

figure
hold on
for i=1:N_temps
E_aux=E_vs_t{i};
plot([ E_aux(2:round(length(E_aux)/20):end-1) E_aux(end)])
end
legend

if use_replicas
    figure
    plot(temps,B,'ob','markersize',2)
    ylabel('$B$','interpreter','latex')
    set(gca,'fontsize',14)
    
%     figure
%     plot(temps,g,'ob','markersize',2)
%     ylabel('$B$','interpreter','latex')
%     set(gca,'fontsize',14)

end

if use_conf && use_replicas
    figure
    plot(temps,Bc,'-ok','markersize',3,'linewidth',2,'Color',[0.5,0.5,0.5],'MarkerEdgeColor','k','MarkerFaceColor','k')
    ylabel('$B$','interpreter','latex')
    print('Figuras\B_SK_3.png','-dpng','-r500')
end

if parallel_tempering
    figure
    semilogy(temps,mP,'.','markersize',20)
    ylabel('Acceptance ratio','interpreter','latex')
    xlabel('$T$','interpreter','latex')
end

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
    clf
    plot(temps,2*sqrt(variances),'.k')
    hold on
    plot(temps,Error_t,'k','linewidth',1.5)
    legend('$2\sqrt{\sigma(T)}$','$dE(n,T)$','interpreter','latex','location','northwest')
    xlabel('$T$','interpreter','latex')
    set(gca,'fontsize',14)
    print('delta_E_100.png','-dpng','-r500')
end

end

for i=1
if figura_variables_vs_anal
% Solucion analitica al modelo de Ising 2D
Tc=(2/log(1+sqrt(2)));   
temps_a=[linspace(T_min,Tc,1000) linspace(Tc,T_max,1000)];% Barrido de temperaturas de la sol. analítica.    
temps_a(length(temps_a)/2)=[];
[E_a, Cv_a, M_a]=Onsager(temps_a,1);    

% Figuras variables
figure
subplot(2,2,1)
hold on
plot([Tc Tc],[min(E) max(E)],'-r','linewidth',1)
plot(temps_a,E_a,'-b','linewidth',2)
plot(temps,E,'ok','markersize',5)
ylabel('$E$','interpreter','latex')
ylim([min(E) max(E)])
pbaspect([1 0.8 1])

subplot(2,2,2)
hold on
plot([Tc Tc],[0 max(Cv)],'-r','linewidth',1)
plot(temps_a,Cv_a,'-b','linewidth',2)
plot(temps,Cv,'ok','markersize',5)
ylabel('$C_V$','interpreter','latex')
ylim([0, max(Cv(2:end))])
pbaspect([1 0.8 1])

subplot(2,2,3)
hold on
plot([Tc Tc],[min(abs(M)) max(abs(M))],'-r','linewidth',1)
plot(temps_a,M_a,'-b','linewidth',2)
plot(temps,abs(M),'ok','markersize',5)
ylabel('$M$','interpreter','latex')
ylim([0 1])
pbaspect([1 0.8 1])

subplot(2,2,4)
hold on
plot([Tc Tc],[0 max(X)],'-r','linewidth',1)
plot(temps,X,'ok','markersize',5)
ylabel('$\chi$','interpreter','latex')
ylim([0, max(X)])
pbaspect([1 0.8 1])
drawnow
print('Figuras\Magnitudes_escalado_Ising_64.png','-dpng','-r500')
end
end

    if estudio_escalado
    Ed(ii,:) = E;
    Md(ii,:) = M;
    M2d(ii,:)= M2;
    Cvd(ii,:)= Cv;
    Xd(ii,:) = X;
    qd(ii,:) = q;
    Bd(ii,:) = Bc;
    
    N(ii)=n;
    end

end

if estudio_escalado
colors=bone(length(N)+1); 
colors(end,:)=[];
colors=colors(end:-1:1,:);
        
f_mags=figure;
subplot(2,2,1)
hold on
for ii=1:size(dim,1)
plot(temps,Ed(ii,:),'-o','markersize',2,'color',colors(ii,:),'linewidth',2)
end
ylabel('$E(T)$','interpreter','latex')
pbaspect([1 0.8 1])

subplot(2,2,2)
hold on
for ii=1:size(dim,1)
plot(temps,Cvd(ii,:),'-o','markersize',2,'color',colors(ii,:),'linewidth',2)
end
ylabel('$C_V$','interpreter','latex')
pbaspect([1 0.8 1])

subplot(2,2,3)
hold on
for ii=1:size(dim,1)
plot(temps,Md(ii,:),'-o','markersize',2,'color',colors(ii,:),'linewidth',2)
end
% ylim([0 1])
ylabel('$M$','interpreter','latex')
pbaspect([1 0.8 1])

subplot(2,2,4)
hold on
for ii=1:size(dim,1)
plot(temps,Xd(ii,:),'-o','markersize',2,'color',colors(ii,:),'linewidth',2,'DisplayName',['$n=$' num2str(N(ii))])
end
ylabel('$\chi$','interpreter','latex')
legend 
pbaspect([1 0.8 1])
print(['Figuras\Magnitudes_escalado_' num2str(modelo) '.png'],'-dpng','-r500')

figure
hold on
for ii=1:size(dim,1)
plot(temps,qd(ii,:),'-o','markersize',2,'color',colors(ii,:),'linewidth',2,'DisplayName',['$n=$' num2str(N(ii))])
end
ylabel('$q_{EA}$','interpreter','latex')
legend
xlabel('$T$','interpreter','latex')  
set(gca,'fontsize',14)
print(['Figuras\q_escalado_' num2str(modelo) '.png'],'-dpng','-r500')

figure
hold on
for ii=1:size(dim,1)
plot(temps,qd(ii,:),'-o','markersize',2,'color',colors(ii,:),'linewidth',2,'DisplayName',['$n=$' num2str(N(ii))])
end
ylabel('$q_{EA}$','interpreter','latex')
legend
xlabel('$T$','interpreter','latex')  
set(gca,'fontsize',14, 'YScale', 'log')
print(['Figuras\q_log_escalado_' num2str(modelo) '.png'],'-dpng','-r500')

figure
hold on
for ii=1:length(N)
plot(temps,Bd(ii,:),'.-','markersize',20,'color',colors(ii,:),'linewidth',2,'DisplayName',['$n=$' num2str(N(ii))])
% Bf=fit(temps',Bd(ii,:)','gauss1');
% fig_fit=plot(Bf)
% set(fig_fit,'color',colors(ii,:),'linewidth',1.5)
end
% legend('$n=36$','','$n=64$','','$n=121$','','$n=256$','','$n=400$','interpreter','latex')  
legend
ylabel('$B$','interpreter','latex')  
xlabel('$T$','interpreter','latex')  
set(gca,'fontsize',14)
print(['Figuras\B_escalado_' num2str(modelo) '.png'],'-dpng','-r500')
 

save('Resultados')
save(['Modelo' num2str(modelo) '_n' num2str(N) '_Nconf' num2str(N_conf) '.dat'],'Ed','Md','Cvd','Xd','qd','Bd','-ascii')

end
toc

% Bd=load('Bd_T_0.6-1.3_n_9_16_25_36_64_121_256.dat');
