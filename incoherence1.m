function mu = incoherence1(U)
K = U*U';
K = K.*K;
mu = max(sum(K))*size(U,1)/size(U,2);
