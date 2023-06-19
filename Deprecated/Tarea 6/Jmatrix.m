function Jm=Jmatrix(n,modelo,seed,RNG,J,J0,r)

if RNG==1
    rand_settings=rng(seed,'twister');
elseif RNG==2
    rand_settings=rng(seed,'multFibonacci');
end

% By default we make a full matrix interaction of everyone with everyone.
if modelo == 4
    pd = makedist('Normal','mu',J0/(n),'sigma',abs(J)/sqrt(n));
else
    pd = makedist('Normal','mu',J0/(n),'sigma',abs(J)); 
end
Jm = random(pd,n,n);

if modelo ==1
        Jm=1;

elseif modelo == 2 % EA 2D
        L = round(n^(1/2));
        J1d=zeros(L,L)+diag(ones(1,L-1),+1)+diag(ones(1,L-1),-1);
        J1d(1,L)=1;
        J1d(L,1)=1;
        I=eye(L,L);
        J2d = kron(I,J1d)+kron(J1d,I);
        J2d(J2d>0)=1;
        Jm=J2d.*Jm; 

elseif modelo == 3 % EA 3D
        L = round(n^(1/3));
        J1d=zeros(L,L)+diag(ones(1,L-1),+1)+diag(ones(1,L-1),-1);
        J1d(1,L)=1;
        J1d(L,1)=1;
        I=eye(L,L);
        J2d = kron(I,J1d)+kron(J1d,I);
        J3d = kron(I,J2d)+kron(J2d,I);
        J3d(J3d>0)=1;
        Jm=J3d.*Jm;  

elseif modelo == 5 % EA 2D + r% of the connections of a square lattice with cross neighbors.

        L = round(n^(1/2));
        J1d=zeros(L,L)+diag(ones(1,L-1),+1)+diag(ones(1,L-1),-1); %+ diag(ones(1,L-2),+2)+diag(ones(1,L-2),-2);
        % Si en 1D incluyo 2 conexiones a segundos vecinos por cada espin, en 2D se traduce a
        % 4 conexiones a segundos vecinos pero a traves de los ejes x e y de la red,
        % es decir, no los segundos vecinos cruzados.

        J1d(1,L)=1;
        J1d(L,1)=1;
        I=eye(L,L);
        J2d = kron(I,J1d) + kron(J1d,I) + kron(J1d,J1d).*rand(n,n);
        J2d(J2d > 1-r) = 1;
        J2d(J2d < 1-r) = 0;
        Jm=J2d.*Jm; 
end

if modelo ~= 1
    % Make it symmetric
    for i=2:n
        for j=1:i-1
            Jm(i,j)=Jm(j,i);
        end
    end
    % No self interaction
    Jm(Jm==diag(Jm))=0; 
end

%         fid = fopen('J_matrix.txt','w');
%         fprintf(fid, '%2.14f\n', Jm);
%         fid = fopen('J_matrix.txt','r') ;
%         Jm = fscanf(fid, '%f', [9,9]) ;
    
end