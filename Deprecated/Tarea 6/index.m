function ind=index(f,i,j)
% Returns the index "ind" of a matrix A, size(A)=[f,c] such that
% A(ind) = A(i,j)
ind = f*(j-1) + i;
end