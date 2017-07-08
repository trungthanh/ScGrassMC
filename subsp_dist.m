function d = subsp_dist(U, V)
R = U\V;
d = norm(U*R-V,'fro');
