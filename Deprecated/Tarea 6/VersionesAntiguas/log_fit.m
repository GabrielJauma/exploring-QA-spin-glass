function [f_fit,formula]=log_fit(f,N,N_0,N_f)
    N_C = [ones(length(N(N_0:N_f)),1) log10(N(N_0:N_f))'];
    T=log10(f(N_0:N_f))';
    b = N_C\T;
    f_fit=(10^b(1))*N.^b(2);
   
    b1=num2str(b(1),'%.2f');
    b2=num2str(b(2),'%.2f');
    formula=['$\log(N_{eq})=$' b1 '+' b2 '$\cdot\log(n)$'];
end