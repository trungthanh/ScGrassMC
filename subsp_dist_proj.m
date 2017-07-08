function d = subsp_dist_proj(U, V)
d = norm(U*U'-V*V','fro');
