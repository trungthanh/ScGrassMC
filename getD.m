% function out = getD_l2(M, X, Y, A, I, J)
% compute min_D |P_M(XDY')-A|_F (D is a diagonal matrix)
% M: sparse mask matrix (m x n) (values will be changed due to the mex trick)
% A: non-zero entris (nnz x 1)
% I,J: row, col indices (nnz x 1)
function out = getD(M, X, Y, A, I, J, lam, reg_type)

[m r] = size(X);
[n r] = size(Y);
l = length(I);

AA = maskmult1(X', Y', I, J);
if strcmp(reg_type, 'l2') == 1
  if lam > 0.0
    S = (AA*AA'+lam*eye(r))\(AA*A');
  else
    S = AA'\A';
  end
elseif strcmp(reg_type, 'l1') == 1
  error('reg_type not yet implemented in getD_l2');
end

out = diag(S);
