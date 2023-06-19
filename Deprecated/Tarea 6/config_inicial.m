function s=config_inicial(f,c,p_dw,seed)
% Crea la configuración inicial de spins del modelo de Ising.
% opt=1 -> Tablero de ajedrez.
% opt=2 -> Aleatorio con probabilidad p_dw € [0,1] de estar abajo (-1).
rand_settings=rng(seed,'twister');
s=ceil( rand(f,c)-(p_dw) );
s(s==0)=-1;
end