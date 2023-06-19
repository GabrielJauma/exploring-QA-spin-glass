function dE=deltaEnergia(s,f_p,c_p,H,J,modelo)

[f,c]=size(s);
dE=0;

if modelo == 1  %Modelo de Ising primeros vecinos
    % Condiciones de contorno peri�dicas.
    if c_p>1 && c_p<c
    L=c_p-1;
    R=c_p+1;
    elseif c_p==1
    L=c;
    R=c_p+1;
    elseif c_p==c
    L=c_p-1;
    R=1;
    end

    if f_p>1 && f_p<f
    U=f_p-1;
    D=f_p+1;
    elseif f_p==1
    U=f;
    D=f_p+1;
    elseif f_p==f
    U=f_p-1;
    D=1;
    end

    dE=2*J*s(f_p,c_p)*(s(f_p,L)+s(f_p,R)+s(U,c_p)+s(D,c_p))-s(f_p,c_p)*2*H;
    
elseif modelo == 2  % Modelo SK primeros vecinos
    % Condiciones de contorno peri�dicas.
    if c_p>1 && c_p<c
    L=c_p-1;
    R=c_p+1;
    elseif c_p==1
    L=c;
    R=c_p+1;
    elseif c_p==c
    L=c_p-1;
    R=1;
    end

    if f_p>1 && f_p<f
    U=f_p-1;
    D=f_p+1;
    elseif f_p==1
    U=f;
    D=f_p+1;
    elseif f_p==f
    U=f_p-1;
    D=1;
    end

     m = f*(c_p-1) + f_p;
     
     dE = 2*s(f_p,c_p)*J(m,f*(L-1) + f_p)*s(f_p,L) ...
         +2*s(f_p,c_p)*J(m,f*(R-1) + f_p)*s(f_p,R) ...
         +2*s(f_p,c_p)*J(m,f*(c_p-1) + U)*s(U,c_p) ...
         +2*s(f_p,c_p)*J(m,f*(c_p-1) + D)*s(D,c_p) -2*H*s(f_p,c_p) ;

elseif modelo >= 3 % Modelo SK rango infinito y modelo EA 3D 
%     m = f*(c_p-1) + f_p;
%     dE = 2*s(m)*J(m,:)*s(:)-2*H*s(f_p,c_p);
 dE = 2*s(f*(c_p-1) + f_p)*J(f*(c_p-1) + f_p,:)*s(:)-2*H*s(f*(c_p-1) + f_p);
end

end