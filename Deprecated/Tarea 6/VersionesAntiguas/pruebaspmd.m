% TEST 1

modelo =3;

f=20;
c=20;
n=f*c;

H = 0;                         % Campo magn�tico externo.
kb= 1;                         % Constante de Botlzmann en unidades en las que vale 1.
J = 1;
J0 = 0;
p_dw = 0.5;                    % La matriz de espines inicial es una matriz aleatoria 
RNG = 1;                       % Generador de n�meros aleatorios:
                               % 1 = Mersenne Twister, 
                               % 2 = Multiplicative Lagged Fibonacci.
seed=6546158;                    

Jm=Jmatrix(f,c,modelo,seed,RNG,J,J0);
s=config_inicial(f,c,p_dw,seed); % Matriz inicial de espines.
cagada=0;

%%
Ei = Energia(s,H,Jm,modelo); %Energia inicial por espin.
for f_p=1:f
    for c_p=1:c

sp=s;
sp(f_p,c_p)=-sp(f_p,c_p);


E(f_p,c_p)=deltaEnergia(s,f_p,c_p,H,Jm,modelo)- n*(Energia(sp,H,Jm,modelo) -Energia(s,H,Jm,modelo));

if abs(E)>1e-10
    'ey'
    cagada=1;
end

    end
end

if ~cagada
    'todo en orden'
end

%% TEST 2
% B=rand(1e8,1);
% C=rand(1e8,1);

for i=1:10

tic 
K1(i)=sum(B.*C);
t1(i)=toc;

tic
K2(i)=dot(B,C);
t2(i)=toc;

tic
K3(i)=B'*C;
t3(i)=toc;

tic
K4(i)=0;
    for j=1:length(B)
    K4(i)=K4(i) + B(j)*C(j);
    end
t4(i)=toc;

end
clc
disp('sum(B.*C)')
fprintf('t_mean = %.5f, t_max = %.5f\n\n',mean(t1),max(t1))
disp('dot(B,C)')
fprintf('t_mean = %.5f, t_max = %.5f\n\n',mean(t2),max(t2))
disp('transpose(B)*C')
fprintf('t_mean = %.5f, t_max = %.5f\n\n',mean(t3),max(t3))
disp('loop')
fprintf('t_mean = %.5f, t_max = %.5f\n',mean(t4),max(t4))

if abs(mean(K1)-mean(K2))<1e-12 && abs(mean(K2)-mean(K3))<1e-12 && abs(mean(K3)-mean(K4))<1e-12
    'Todo en orden'
end


%% Test 3
t=[];
workers=[1 2 4 6 8 12 20];

for i =workers
i    
tic
spmd
       
if labindex<=i
    for j=1:10
    A=rand(5e3,5e3);
    B=rand(5e3,1);
    x=B\A;
    end
end
      
end
t=[t toc];

end
plot(workers,t./workers)


%% Test 3 part 2
t2=[]
for i = workers
i    
tic
for k=1:i    
    for j=1:10
    A=rand(5e3,5e3);
    B=rand(5e3,1);
    x=B\A;
    end
end
t2=[t2 toc];

end
hold on
plot(workers,t2./workers)



