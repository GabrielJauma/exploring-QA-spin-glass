function dE=deltaEnergiaIsing(s,f_p,c_p,J,H)
%% Condiciones de contorno periódicas.
[f,c]=size(s);
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

end