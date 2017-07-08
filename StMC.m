function [U D V hist] = StMC(K, A_K, r, varargin)

params     = get_varargin(varargin);
maxit      = get_params(params, 'maxit', 100);
tol        = get_params(params, 'tol', 1.e-4);
beta_type  = get_params(params, 'beta_type', 'P-R');     % 'P-R', 'F-R' or 'Steep'
grad_type  = get_params(params, 'grad_type', 'scaled');  % 'canon', 'scaled', 'sqrtscaled'
reg_type   = get_params(params, 'reg_type', 'l2');
init_type  = get_params(params, 'init_type', 'svd');
orth_value = get_params(params, 'orth_value', 0.1);
regval     = get_params(params, 'regval', 0.0);
verbose    = get_params(params, 'verbose', 0);

% make A_K a row vector
if size(A_K,1)>1
  A_K = A_K';
end

num_known = length(A_K);
[m n] = size(K);
[I J] = find(K);
MASK = sparse(I,J,ones(num_known,1),m,n);
I = int32(I);
J = int32(J);

% initialize U and V
if strcmp(init_type, 'svd') == 1
  [x.U x.D x.V] = svds(sparse(double(I),double(J),A_K,m,n), r);
elseif strcmp(init_type, 'given')
  x.U = params.U;
  x.V = params.V;
end

% compute D
if isfield(params,'D')
  x.D = params.D;
else
  x.D = getoptS_mex(MASK, x.U, x.V, A_K, I, J, regval);
  [R1 x.D R2] = svd(x.D); x.U = x.U*R1; x.V = x.V*R2; % diagonalize
end

start_time = tic;

x.UD = x.U*x.D;
x.X = maskmult(x.UD',x.V',I,J);
x.E = x.X-A_K;
x.Emat = sparse(double(I),double(J),x.E,m,n);

itc = 1;
xc = x; 
fc = F(xc, regval, reg_type);
gc = grad(xc, grad_type);
ip_gc = ip(xc,gc,gc,grad_type);
dir = scaleTxM(gc,-1); % first search-dir is steepest gradient

ithist     = zeros(maxit, 2);
step_sizes = zeros(maxit, 1);
beta = 0;
total_time = 0;
for itc=1:maxit

  if itc>1, start_time = tic; end

  % exact line search and then back-tracking Armijo search
  step_sizes(itc) = exact_search_onR1(I, J, xc, dir);
  [xc_new,fc,succ,numf_a1,iarm1,step_sizes(itc)] = armijo_search(A_K,MASK,I,J,xc,fc,dir,step_sizes(itc),grad_type,regval,reg_type);
  %[xc_new fc step_sizes(itc)] = backtrack(A_K,MASK,I,J,xc,fc,dir,regval,reg_type);

  % gradients(x_new)
  gc_new = grad(xc_new,grad_type);
  ip_gc_new = ip(xc_new,gc_new,gc_new,grad_type);
  
  % test for convergence
  if sqrt(2*fc/num_known) < tol
    ithist(itc,1) = fc; 
    total_time = total_time + toc(start_time);
    ithist(itc,2) = total_time;
    if verbose
      fprintf('Iter %d - Objective: %f\n', itc, ithist(itc,1));
    end
    break;
  end

  % transport old gradient and search direction
  gc_old = transport(A_K,MASK,I,J,xc,gc,xc_new,1,grad_type,reg_type);
  dir = transport(A_K,MASK,I,J,xc,dir,xc_new,1,grad_type,reg_type);
  %gc_temp.U = step_sizes(itc)*gc.U; gc_temp.V = step_sizes(itc)*gc.V;
  %gc_old = transport(A_K,MASK,I,J,xc,gc,gc_temp,0,grad_type,reg_type);
  %dir = transport(A_K,MASK,I,J,xc,dir,gc_temp,0,grad_type,reg_type);
  
  % we test how orthogonal the previous gradient is with
  % the current gradient, and possibly reset the to the gradient
  orth_grads = ip(xc_new,gc_old,gc_new,grad_type)/ip_gc_new;
  
  % compute beta
  if strcmp(beta_type, 'Steep') %| orth_grads >= orth_value
    dir = plusTxM(gc_new, dir, -1, 0);
  elseif strcmp(beta_type, 'F-R')  % Fletcher-Reeves
    beta = ip_gc_new/ip_gc;
    dir = plusTxM(gc_new, dir, -1, beta);
  elseif strcmp(beta_type, 'P-R')  % Polak-Ribiere
    diff = plusTxM(gc_new, gc_old, 2, -1); % grad(new) - transported(grad(current))
    ip_diff = ip(xc_new, gc_new, diff, grad_type);
    beta = ip_diff / ip_gc;
    beta = max(0,beta);
    dir = plusTxM(gc_new, dir, -1, beta);
  elseif strcmp(beta_type, 'H-Z')
    yk_c = plusTxM(gc_new, gc_old, 1, -1);
    dk_c = transport(A_K, MASK, I, J, xc,dir, xc_new, 1, grad_type, reg_type);
    temp = plusTxM(yk_c,dk_c,1,-2*ip(xc_new,yk_c,yk_c, grad_type)/ip(xc_new,yk_c,dk_c,grad_type));
    beta = 1/ip(xc_new,yk_c,dk_c,grad_type)*ip(xc_new,temp,gc_new,grad_type);
    ct = -1/(sqrt(ip(xc_new,dk_c,dk_c,grad_type))*min(0.01,sqrt(ip_gc)));
    beta = max(ct,beta);
    dir = plusTxM(gc_new, dir, -1, beta);    
  end

  % update new to current
  gc = gc_new;
  ip_gc = ip_gc_new;
  xc = xc_new;

  ithist(itc,1) = fc; 
  total_time = total_time + toc(start_time);
  ithist(itc,2) = total_time;
  if verbose
    fprintf('Iter %d - Objective: %f\n', itc, ithist(itc,1));
  end

  if itc>5 & isfield(params,'tol_reschg')
    if abs(1-sqrt(ithist(itc,1)/ithist(itc-1,1))) < params.tol_reschg
      break;
    end
  end

end

U = xc_new.U;
D = xc_new.D;
V = xc_new.V;
hist.obj = ithist(1:itc,1);
hist.time = ithist(1:itc,2);


%%---------- compute function F
function out = F(x, regval, reg_type)
if nargin < 3
  reg_type = 'l1';
end
out = 0.5*(norm(x.E)^2);
if regval>0.0
  switch reg_type
  case 'l1'
    out = out + regval*norm(diag(x.D),1);
  case 'l2'
    out = out + regval*(norm(diag(x.D))^2);
  otherwise
    error('reg_type not yet implemented');
  end
end


%%---------- symmetrize a function
function S = symm(A)
S = 0.5*(A+A');


%%---------- compute gradients
function g = grad(x, grad_type)
switch grad_type
case 'canon'
  VD = x.V*x.D;
  EVD = x.Emat*VD;
  EtUD = (x.UD'*x.Emat)';
  UtEVD = x.U'*EVD;
  VtEtUD = x.V'*EtUD;
  g.U = EVD - x.U*symm(UtEVD);
  g.V = EtUD - x.V*symm(VtEtUD);
  if 0
    g.U = g.U*inv(x.D^2);
    g.V = g.V*inv(x.D^2);
  end
otherwise
  error('grad_type not yet implemented in grad');
end


%%---------- compute inner product <h1,h2> at point x
%% note: we can use different ip at different points
function i = ip(x, h1, h2, grad_type)
switch grad_type
case {'canon'}
  i = trace(h1.U'*h2.U) + trace(h1.V'*h2.V);
  if 0
    D = x.D^2;
    i = trace(D*h1.U'*h2.U) + trace(D*h1.V'*h2.V);
  end
otherwise
  error('grad_type not yet implemented in ip');
end


%%---------- scaling on tangent space
function h = scaleTxM(h, a)
h.U = a*h.U;
h.V = a*h.V;


% add 2 tangent vectors in the same tangent space
%  h = a1*h1 + a2*h2  a1,a2 reals
function h = plusTxM(h1,h2,a1,a2)
h.U = a1*h1.U + a2*h2.U;
h.V = a1*h1.V + a2*h2.V;


%  Retract a point x+t*h (minimal distance)
%  z = retract(sys,x,h,t) retracts x+t*h to the manifold where sys is a system
%     x is a point of sys,
%     h is a tangent vector of TxM,
%     t is a distance (scalar).
function z = retract(A_K, K, I, J, x, h, t, is_dir, regval, reg_type)
[m n] = size(K);
[z.U Ru] = qr(x.U + t*h.U,0);
[z.V Rv] = qr(x.V + t*h.V,0);
if is_dir % if this is a direction, no need to update D, X, E, etc
  return;
end
z.D = getD(K, z.U, z.V, A_K, I, J, regval, reg_type);
z.UD = z.U*z.D;
z.X = maskmult(z.UD',z.V',I,J);
z.E = z.X-A_K;
z.Emat = sparse(double(I), double(J), z.E, m, n);


%% rhoskew used in vector transport of Stiefel manifold
function out = rhoskew(B)
out = tril(B,-1) - triu(B,1);


%% transport the vector (x,h) along the direction of d
% type == 0: transport(x,h,d)
%    we transport by projecting h orth. on R_x(d)
% type == 1: transport(x,h,z)
%    we transport by projecting h orth. on z
% returns ht, the transported h
function ht = transport(A_K, K, I, J, x, h, d, type, grad_type, reg_type)
%% z is the foot of ht
if type==0    
  z = retract(A_K, K, I, J, x, d, 1, 1, 0.0, reg_type);
  switch grad_type
  case {'canon'}
    Temp = inv(z.U'*(x.U+d.U));
    ht.U = z.U*rhoskew(z.U'*h.U*Temp) + h.U*Temp - z.U*(z.U'*h.U*Temp);
    Temp = inv(z.V'*(x.V+d.V));
    ht.V = z.V*rhoskew(z.V'*h.V*Temp) + h.V*Temp - z.V*(z.V'*h.V*Temp);
    if 0
      ht.U = ht.U*inv(z.D^2);
      ht.V = ht.V*inv(z.D^2);
    end
  end
else % type =1
  z = d;
  switch grad_type
  case {'canon'}
    ht.U = h.U - 0.5*z.U*symm(z.U'*h.U);
    ht.V = h.V - 0.5*z.V*symm(z.V'*h.V);
    if 0
      ht.U = ht.U*inv(z.D^2);
      ht.V = ht.V*inv(z.D^2);
    end
  end
end

% the given vector is (x,h) and is transported onto d
ip_old = ip(x, h, h, grad_type);


ip_new = ip(z, ht, ht, grad_type);
ht = scaleTxM(ht, ip_old/ip_new);


%%---------- backtracking search with Armijo condition
function [xt ft t] = backtrack(A_K, K ,I, J, xc, fc, gc, regval, reg_type)
normg2 = norm(gc.U,'fro')^2 + norm(gc.V,'fro')^2;
t = 1.e+2;
while t>1.e-8
  xt.U = xc.U + t*gc.U;
  xt.V = xc.V + t*gc.V;
  switch reg_type
  case 'l1'
    ft = 0.5*(norm(maskmult(xc.D*xt.U',xt.V',I,J)-A_K)^2) + regval*norm(diag(xc.D),1);
  case 'l2'
    ft = 0.5*(norm(maskmult(xc.D*xt.U',xt.V',I,J)-A_K)^2) + regval*(norm(diag(xc.D))^2);
  otherwise
    error('reg_type not yet implemented in backtrack');
  end
  if ft-fc < -.5*t*normg2
    break;
  end
  t = t/2;
end
xt = retract(A_K,K,I,J,xc,gc,t,0,regval,reg_type);
ft = F(xt, regval, reg_type);


% Exact line search in the direction of dir on the tangent space of x
% !! so NOT the retracted curve, only use as guess !!
% x, current point
% dir is search direction
% compute tmin such that f(x + tmin*dir) is minimized
% this only minimizes \|P_Omega(A - xnew.U*D*xnew.V')\|^2 (assuming the same D at xnew)
function tmin = exact_search_onR1(I, J, x, dir)
f0_omega = x.E;
dirUxSt = (dir.U*x.D)';
f1_omega = maskmult(dirUxSt, x.V', I, J);
f1_omega = f1_omega + maskmult(x.UD', dir.V', I, J);
f2_omega = maskmult(dirUxSt, dir.V', I, J);

a1 = 2*sum(f0_omega.*f1_omega);
a2 = norm(f1_omega)^2 + 2*sum(f2_omega.*f0_omega);
a3 = 2*sum(f2_omega.*f1_omega);
a4 = norm(f2_omega)^2;

ts = roots([4*a4 3*a3 2*a2 a1]);
ind_real = find(ts==real(ts));
ts = ts(ind_real);
ind_real = find(ts>0);
if length(ind_real)>1
  tmin = max(ts(ind_real));
elseif length(ind_real)==0
  disp('no positive root');
  tmin = 1.0; % TODO
else
  tmin = ts(ind_real);
end


% Armijo line search with polynomial interpolation
% Adapted from Boumal and Absil, 2011
% Adapted from steep by C. T. Kelley, Dec 20, 1996
% xc, current point
% fc = F(sys,xc)
% gc is search direction
%
% returns: xt,ft: Armijo point and F(sys,xt)
%          succ = true if success
%          numf: nb of F evals
%          iarm: nb of Armijo backtrackings
function [xt,ft,succ,numf,iarm,lambda] = armijo_search(A_K, K, I, J, xc, fc, gc, lambda, grad_type, regval, reg_type)
alp = 1e-4; bhigh=.5; blow=.1; MAX_IARM = 5; numf = 0;

% trial evaluation at full step
xt = retract(A_K, K, I, J, xc, gc, lambda, 0, regval, reg_type);
ft = F(xt, regval, reg_type);
numf = numf+1; iarm = 0;

fgoal = fc - alp*lambda*ip(xc,gc,gc,grad_type);
% polynomial line search
q0=fc; qp0=-ip(xc,gc,gc,grad_type); lamc=lambda; qc=ft;
while(ft > fgoal)    
  iarm = iarm + 1;
  if iarm==1
    lambda=polymod(q0, qp0, lamc, qc, blow, bhigh);
  else
    lambda=polymod(q0, qp0, lamc, qc, blow, bhigh, lamm, qm);
  end
  qm=qc; lamm=lamc; lamc=lambda;
  xt = retract(A_K, K, I, J, xc, gc, lambda, 0, regval, reg_type);
  ft = F(xt,regval,reg_type);
  numf = numf+1; qc=ft;
  if(iarm > MAX_IARM)
    succ = false;
    return
  end
  fgoal = fc-alp*lambda*ip(xc,gc,gc,grad_type);
end
succ = true;


% C. T. Kelley, Dec 29, 1997
% This code comes with no guarantee or warranty of any kind.

% function [lambda]=polymod(q0, qp0, qc, blow, bhigh, qm)
% Cubic/quadratic polynomial linesearch
% Finds minimizer lambda of the cubic polynomial q on the interval
% [blow * lamc, bhigh * lamc] such that
%
% q(0) = q0, q'(0) = qp0, q(lamc) = qc, q(lamm) = qm
% 
% if data for a cubic is not available (first stepsize reduction) then
% q is the quadratic such that
% 
% q(0) = q0, q'(0) = qp0, q(lamc) = qc
function [lplus]=polymod(q0, qp0, lamc, qc, blow, bhigh, lamm, qm)
lleft=lamc*blow; lright=lamc*bhigh; 
if nargin == 6
  % quadratic model (temp hedge in case lamc is not 1)
  lplus = - qp0/(2 * lamc*(qc - q0 - qp0) );
  if lplus < lleft lplus = lleft; end
  if lplus > lright lplus = lright; end
else
  % cubic model
  a=[lamc^2, lamc^3; lamm^2, lamm^3];
  b=[qc; qm]-[q0 + qp0*lamc; q0 + qp0*lamm];
  c=a\b;
  lplus=(-c(1)+sqrt(c(1)*c(1) - 3 *c(2) *qp0))/(3*c(2));
  if lplus < lleft lplus = lleft; end
  if lplus > lright lplus = lright; end
end

