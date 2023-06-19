
n=6^2;
L = round(n^(1/2));
J1d=zeros(L,L)+diag(ones(1,L-1),+1)+diag(ones(1,L-1),-1); %+ diag(ones(1,L-2),+2)+diag(ones(1,L-2),-2);
% Si en 1D incluyo 2 conexiones a segundos vecinos por cada espin, en 2D se traduce a
% 4 conexiones a segundos vecinos pero a traves de los ejes x e y de la red,
% es decir, no los segundos vecinos cruzados.

J1d(1,L)=1;
J1d(L,1)=1;
C=ones(L,L);
C(2:2:L,1:2:L) = 0;
C(1:2:L,2:2:L) = 0;
I=eye(L,L);
J2d = kron(I,J1d) + kron(J1d,I)+ kron(J1d,J1d) ;


% G=graph(J2d);
% plot(G,'Layout','subspace')
% figure
% pcolor(J2d)
% colormap(flipud(gray))
% axis equal
% pbaspect([1 1 1])

Cc = repmat({C}, 1, 3);  
Cb = blkdiag(Cc{:});
figure
pcolor(Cb)
colormap(flipud(gray))
axis equal
pbaspect([1 1 1])


%%
EdgeTable = table([1 2; 1 3; 2 4; 3 4],'VariableNames',{'EndNodes'});
G = graph(EdgeTable);

L=2;

EdgeTable = table(L+[ 1 3; 2 4; 3 4],'VariableNames',{'EndNodes'});
G = addedge(G,EdgeTable);

EdgeTable = table(L+[ 1 3; 2 4; 3 4],'VariableNames',{'EndNodes'});
G = addedge(G,EdgeTable);

plot(G)
