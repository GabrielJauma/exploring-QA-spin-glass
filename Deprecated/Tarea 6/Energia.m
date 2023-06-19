function E=Energia(s,H,J,modelo)
% Calcula la energ�a por espin de una cierta configuraci�n dada por una
% matriz de espines, s.
       E=0;
       [f,c]=size(s);
       
if modelo == 1 %Modelo de Ising
   for i = 2:f-1
       for j = 2:c-1
          E=E + ( -J*s(i,j)*(s(i-1,j)+s(i+1,j)+s(i,j-1)+s(i,j+1)) + H*s(i,j) );
       end
   end
       for j = 1:c
       i=1; %Fila 1
           if j==1
           E=E + ( -J*s(i,j)*(s(end,j)+s(i+1,j)+s(i,end)+s(i,j+1)) + H*s(i,j) );
           elseif j==c
           E=E + ( -J*s(i,j)*(s(end,j)+s(i+1,j)+s(i,j-1)+s(i,1)) + H*s(i,j) );
           else
           E=E + ( -J*s(i,j)*(s(end,j)+s(i+1,j)+s(i,j-1)+s(i,j+1)) + H*s(i,j) );
           end
       i=f; %Fila final
           if j==1
           E=E + ( -J*s(i,j)*(s(i-1,j)+s(1,j)+s(i,end)+s(i,j+1)) + H*s(i,j) );
           elseif j==c
           E=E + ( -J*s(i,j)*(s(i-1,j)+s(1,j)+s(i,j-1)+s(i,1)) + H*s(i,j) );
           else
           E=E + ( -J*s(i,j)*(s(i-1,j)+s(1,j)+s(i,j-1)+s(i,j+1)) + H*s(i,j) );
           end
       end
       for i = 2:f-1
       j=1; %Columna 1 (sin extremos [1,1] ni [end,1] )
           E=E + ( -J*s(i,j)*(s(i-1,j)+s(i+1,j)+s(i,end)+s(i,j+1)) + H*s(i,j) );
       j=c; %Columna final (sin extremos [1,end] ni [end,end] )
           E=E + ( -J*s(i,j)*(s(i-1,j)+s(i+1,j)+s(i,j-1)+s(i,1)) + H*s(i,j) );
       end
       
             
elseif modelo == 2 % Modelo EA a primeros vecinos
   for i = 2:f-1
       for j = 2:c-1
          %E=E + ( -J*s(i,j)*(s(i-1,j)+s(i+1,j)+s(i,j-1)+s(i,j+1)) + H*s(i,j) );
          E = E -s(i,j)*J(index(f,i,j),index(f,i-1,j))*s(i-1,j) ...
                -s(i,j)*J(index(f,i,j),index(f,i+1,j))*s(i+1,j) ...
                -s(i,j)*J(index(f,i,j),index(f,i,j-1))*s(i,j-1) ...
                -s(i,j)*J(index(f,i,j),index(f,i,j+1))*s(i,j+1) + H*s(i,j) ;
       end
   end
       for j = 1:c
       i=1; %Fila 1
           if j==1
           %E=E + ( -J*s(i,j)*(s(f,j)+s(i+1,j)+s(i,c)+s(i,j+1)) + H*s(i,j) );
           E = E -s(i,j)*J(index(f,i,j),index(f,f,j))*s(f,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i+1,j))*s(i+1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,c))*s(i,c) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j+1))*s(i,j+1) + H*s(i,j) ;

           elseif j==c
          % E=E + ( -J*s(i,j)*(s(f,j)+s(i+1,j)+s(i,j-1)+s(i,1)) + H*s(i,j) );
           E = E -s(i,j)*J(index(f,i,j),index(f,f,j))*s(f,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i+1,j))*s(i+1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j-1))*s(i,j-1) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,1))*s(i,1) + H*s(i,j) ;
             
           else
           %E=E + ( -J*s(i,j)*(s(f,j)+s(i+1,j)+s(i,j-1)+s(i,j+1)) + H*s(i,j) );
           E = E -s(i,j)*J(index(f,i,j),index(f,f,j))*s(f,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i+1,j))*s(i+1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j-1))*s(i,j-1) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j+1))*s(i,j+1) + H*s(i,j) ;
           end
           
       i=f; %Fila final
           if j==1
           %E=E + ( -J*s(i,j)*(s(i-1,j)+s(1,j)+s(i,c)+s(i,j+1)) + H*s(i,j) );
           E = E -s(i,j)*J(index(f,i,j),index(f,i-1,j))*s(i-1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,1,j))*s(1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,c))*s(i,c) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j+1))*s(i,j+1) + H*s(i,j) ;
           
           elseif j==c
           %E=E + ( -J*s(i,j)*(s(i-1,j)+s(1,j)+s(i,j-1)+s(i,1)) + H*s(i,j) );
           E = E -s(i,j)*J(index(f,i,j),index(f,i-1,j))*s(i-1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,1,j))*s(1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j-1))*s(i,j-1) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,1))*s(i,1) + H*s(i,j) ;
           
           else
           %E=E + ( -J*s(i,j)*(s(i-1,j)+s(1,j)+s(i,j-1)+s(i,j+1)) + H*s(i,j) );
           E = E -s(i,j)*J(index(f,i,j),index(f,i-1,j))*s(i-1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,1,j))*s(1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j-1))*s(i,j-1) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j+1))*s(i,j+1) + H*s(i,j) ;
           end
       end
       
       for i = 2:f-1
       j=1; %Columna 1 (sin extremos [1,1] ni [end,1] )
          % E=E + ( -J*s(i,j)*(s(i-1,j)+s(i+1,j)+s(i,c)+s(i,j+1)) + H*s(i,j) );
           E = E -s(i,j)*J(index(f,i,j),index(f,i-1,j))*s(i-1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i+1,j))*s(i+1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,c))*s(i,c) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j+1))*s(i,j+1) + H*s(i,j) ;
             
       j=c; %Columna final (sin extremos [1,end] ni [end,end] )
           %E=E + ( -J*s(i,j)*(s(i-1,j)+s(i+1,j)+s(i,j-1)+s(i,1)) + H*s(i,j) );
           E = E -s(i,j)*J(index(f,i,j),index(f,i-1,j))*s(i-1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i+1,j))*s(i+1,j) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,j-1))*s(i,j-1) ...
                 -s(i,j)*J(index(f,i,j),index(f,i,1))*s(i,1) + H*s(i,j) ;
       end

elseif modelo>=3 %Modelo SK de rango infinito o modelo EA 3D
    E = -s(:)'*J*s(:) + H*sum(s(:));
end

% Energ�a POR cada espin.
E=E/(2*f*c);
end