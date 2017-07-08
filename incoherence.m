function mu = incoherence(U)
U = U';
mu = max(sum(U.*U))*size(U,1)/size(U,2);
mu = mu*sqrt(size(U,2))/sqrt(size(U,1));
