function out = colgen_lmi(N, opts)
% COLGEN_LMI  Column-generation solver for the ORIGINAL forward-Euler LMI, Eq. (24):
%     min gamma  s.t.  I <= P_j <= gamma*I              (j = 1..K)
%                      (P_{j+1}-P_j)/dt + A_j'*P_j + P_j*A_j <= 0   (j = 1..K-1)
% with the SAME A_j, K, dt and indexing as transcient_growth_new.m.
%
% Idea: P_j = I + U_j*S_j*U_j' with a small basis U_j (n x r_j).
%   * Upper bound: the restricted problem in S_j only (exact Schur-reduced LMIs)
%     gives a feasible P, so its gamma is a valid upper bound.
%   * Lower bound: the restricted duals are lifted to the full space to build
%     Lambda_j; a rigorous lower bound on gamma* is computed every round.
%   * Pricing: negative eigenvectors of Lambda_j are added to U_j; repeat.
% Lower bound is rigorous: sum tr(Lambda+) / (sum tr(Omega) + sum tr(Lambda-)).
% Stops when (UB - LB)/UB < opts.tol (default 1e-5, safely inside 4 significant figures).
%
% Usage:  out = colgen_lmi(12);           % expected gamma ~ 85.66
%         out = colgen_lmi(10);           % expected ~ 178.0
%         out = colgen_lmi(20);           % ~32.17 (upper bound; certificate still converging)
% Requires YALMIP + MOSEK on the path.

if nargin < 1 || isempty(N), N = 12; end
if nargin < 2, opts = struct(); end
d = struct('kx',1.2,'kz',0,'Re',500,'kappa',0.1,'t0',0,'T',30,'dt',1,'NN',100, ...
           'r0',3,'addmax',2,'tol',1e-5,'maxit',40,'rho',1e4,'solver','mosek','verbose',0,'prune',false,'addrel',0);
f = fieldnames(d);
for k = 1:numel(f), if ~isfield(opts,f{k}), opts.(f{k}) = d.(f{k}); end, end

tic;
A = build_A(N, opts);  K = numel(A);  Kc = K-1;  n = size(A{1},1);  dt = opts.dt;
H = cell(K,1); for j = 1:K, H{j} = A{j}+A{j}'; H{j} = (H{j}+H{j}')/2; end
fprintf('N = %d: %d matrices A_j (%dx%d), %d dynamics LMIs\n', N, K, n, n, Kc);

% ---- warm start (backward Lyapunov recursion, feasible) -------------------
X = cell(K,1); X{K} = zeros(n);
for j = Kc:-1:1
    [Vh,Dh] = eig(H{j}); Hp = Vh*diag(max(real(diag(Dh)),0))*Vh'; Hp = (Hp+Hp')/2;
    % (I/(2dt) - A_j')X_j + X_j(I/(2dt) - A_j) = X_{j+1}/dt + H_j^+   =>   M_j = H_j - H_j^+ <= 0
    X{j} = sylvester(eye(n)/(2*dt) - A{j}', eye(n)/(2*dt) - A{j}, X{j+1}/dt + Hp);
    X{j} = (X{j}+X{j}')/2;
end
gw = 1; for j = 1:K, gw = max(gw, 1+max(real(eig(X{j})))); end
fprintf('warm start gamma = %.4f\n', gw);
U = cell(K,1); for j = 1:K, U{j} = topvecs(X{j}, opts.r0); end

hist = [];
for it = 1:opts.maxit
    % ---- exact Schur reduction of each dynamics LMI ------------------------
    tb = tic;
    W = cell(Kc,1); T = W; R = W;
    for j = 1:Kc, [W{j},T{j},R{j}] = reduce_step(H{j}, A{j}, U{j}, U{j+1}); end

    % ---- restricted problem (YALMIP) ---------------------------------------
    yalmip('clear');
    % Complex Hermitian LMIs are written in the equivalent REAL form
    % X >= 0  <=>  [Re X, -Im X; Im X, Re X] >= 0, because YALMIP does not
    % return MOSEK's duals for complex constraints (they come back NaN).
    % Fast model building: each S_j = sum_k x_k E_k over a Hermitian basis, and
    % every constraint is precomputed NUMERICALLY as (constant + G*x), so YALMIP
    % never does symbolic matrix products (that was the bottleneck).
    x = cell(K,1); Eb = cell(K,1); Gs = cell(K,1);
    for j = 1:K
        r = size(U{j},2); Eb{j} = hbasis(r); x{j} = sdpvar(r*r,1);
        Gs{j} = embvec(Eb{j}, r);                       % vec(emb(S_j)) = Gs*x
    end
    g = sdpvar(1,1); s = sdpvar(Kc,1);
    Clo = []; Cup = []; Cdyn = [];
    for j = 1:K
        r = size(U{j},2); I2 = eye(2*r);
        Clo = [Clo, reshape(Gs{j}*x{j}, 2*r, 2*r) >= 0];                 %#ok<AGROW>
        Cup = [Cup, (g-1)*I2 - reshape(Gs{j}*x{j}, 2*r, 2*r) >= 0];      %#ok<AGROW>
    end
    for j = 1:Kc
        m = size(W{j},2);
        B1 = W{j}'*U{j}; B2 = W{j}'*U{j+1}; C1 = W{j}'*(A{j}'*U{j});
        % L_r = (B2 S2 B2' - B1 S1 B1')/dt + C1 S1 B1' + B1 S1 C1'   (vec(BEC') = kron(conj(C),B) vec(E))
        K1 = (-kron(conj(B1),B1)/dt + kron(conj(B1),C1) + kron(conj(C1),B1)) * Eb{j};
        K2 = (kron(conj(B2),B2)/dt) * Eb{j+1};
        G1 = embvec(K1, m); G2 = embvec(K2, m);
        ER = [real(R{j}), -imag(R{j}); imag(R{j}), real(R{j})];
        Mdyn = s(j)*eye(2*m) - ER - reshape(G1*x{j} + G2*x{j+1}, 2*m, 2*m);
        Cdyn = [Cdyn, Mdyn >= 0];                                         %#ok<AGROW>
    end
    tbuild = toc(tb);
    sol = optimize([Clo, Cup, Cdyn, s >= 0], g + opts.rho*sum(s), ...
                   sdpsettings('solver',opts.solver,'verbose',opts.verbose));
    if sol.problem ~= 0, fprintf('   (solver note: %s)\n', sol.info); end
    gam = value(g); smax = max(value(s));
    Sval = cell(K,1);
    for j = 1:K, r = size(U{j},2); Sval{j} = reshape(Eb{j}*value(x{j}), r, r); end

    % ---- lift the duals to the full space ----------------------------------
    Y = cell(Kc,1);
    for j = 1:Kc, y = psdpart(unemb(dual(Cdyn(j)))); Y{j} = T{j}*y*T{j}'; end
    LB = 0; LBpos = 0; LBneg = 0; trOm = 0; minLam = inf; nadd = 0; Unew = U;
    tp = tic;  LV = cell(K,1); LL = cell(K,1);
    for j = 1:K
        om = psdpart(unemb(dual(Cup(j)))); trOm = trOm + real(trace(om));
        Lam = U{j}*om*U{j}';
        if j >= 2, Lam = Lam + Y{j-1}/dt; end
        if j <= Kc, Lam = Lam - Y{j}/dt + A{j}*Y{j} + Y{j}*A{j}'; end
        Lam = (Lam+Lam')/2;
        [Vl,Dl] = eig(Lam); lam = real(diag(Dl)); [lam,ix] = sort(lam); LV{j} = Vl(:,ix); LL{j} = lam;
        LB = LB + sum(lam); minLam = min(minLam, lam(1));
        LBpos = LBpos + sum(max(lam,0)); LBneg = LBneg + sum(max(-lam,0));
    end
    % Pricing: add the most negative eigenvectors of Lambda_j. With addrel > 0,
    % only directions with lambda < addrel * (most negative lambda over ALL j)
    % are added, so steps that are already nearly optimal do not grow.
    for j = 1:K
        lam = LL{j};
        thr = -1e-7*max(1,max(abs(lam)));
        if opts.addrel > 0, thr = min(thr, opts.addrel*minLam); end
        neg = find(lam < thr, opts.addmax);
        base = U{j};
        if opts.prune && smax < 1e-7*max(1,gam)
            Sj = (Sval{j}+Sval{j}')/2; [Vs,Ds] = eig(Sj); ds = real(diag(Ds));
            keep = ds > 1e-6*max(1,gam); if ~any(keep), [~,keep] = max(ds); end
            base = U{j}*Vs(:,keep);
        end
        Unew{j} = orth([base, LV{j}(:,neg)]);
        nadd = nadd + (size(Unew{j},2) - size(base,2));
    end
    tprice = toc(tp);
    % Rigorous lower bound. The multipliers Y_j, Omega_j are projected to be PSD,
    % and Lambda_j is DEFINED by the stationarity equation, so stationarity in P
    % holds exactly. The negative part of each Lambda_j is moved into Omega_j,
    % and everything is divided by the actual sum(trace(Omega)) (solvers only
    % satisfy sum(trace(Omega)) = 1 approximately). Weak duality: gamma* >= LBcert.
    LBcert = LBpos/(trOm + LBneg);
    gap = (gam - LBcert)/gam;
    rk = cellfun(@(u) size(u,2), U);
    if smax > 1e-7*max(1,gam), tag = 'NOT yet feasible'; else, tag = 'valid UB'; end
    fprintf(['it %2d: gamma = %.6f [%s, max slack %.1e] | certified LB = %.6f (min eig Lambda %.1e) | ' ...
             'gap %.2e | ranks %d-%d | +%d dirs | %.0fs\n'], ...
            it, gam, tag, smax, LBcert, minLam, gap, min(rk), max(rk), nadd, toc);
    msz = cellfun(@(w) size(w,2), W);
    fprintf('       time this round: build %.1fs | YALMIP %.1fs | solver %.1fs | pricing %.1fs | largest reduced LMI %dx%d (n = %d)\n', ...
            tbuild, sol.yalmiptime, sol.solvertime, tprice, max(msz), max(msz), n);
    hist = [hist; it, gam, smax, LB, minLam]; %#ok<AGROW>
    Usol = U;    % basis that goes with Sval (U itself may grow below)
    % converged: gap below tol AND bounds consistent (LB > UB means the solver's
    % duals are not accurate enough yet, so keep refining)
    if smax < 1e-7*max(1,gam) && gap < opts.tol && gap > -1e-6, break; end
    if nadd == 0, break; end   % pricing found no new direction
    U = Unew;
end

% ---- independent check on the FULL constraints ------------------------------
P = cell(K,1); minP = inf; maxP = 0;
for j = 1:K
    Sj = (Sval{j}+Sval{j}')/2; P{j} = eye(n) + Usol{j}*Sj*Usol{j}'; P{j} = (P{j}+P{j}')/2;
    e = real(eig(P{j})); minP = min(minP, min(e)); maxP = max(maxP, max(e));
end
worst = -inf;
for j = 1:Kc
    M = (P{j+1}-P{j})/dt + A{j}'*P{j} + P{j}*A{j}; M = (M+M')/2;
    worst = max(worst, max(real(eig(M))));
end
fprintf('\nFINAL  gamma (upper bound)   = %.6f\n', gam);
fprintf('       certified lower bound = %.6f\n', LBcert);
fprintf('       relative gap          = %.2e  (target < %.0e)\n', gap, opts.tol);
fprintf('       full check: max eig M_j = %.2e, eig P in [%.6f, %.6f]\n', worst, minP, maxP);
fprintf('       total time %.1f s\n', toc);
out = struct('gamma',gam,'LB',LBcert,'gap',gap,'maxeigM',worst,'P',{P},'U',{Usol},'hist',hist);
end

% =========================================================================
function [W, T, R] = reduce_step(H, A, U1, U2)
% W spans U_j, U_{j+1}, A_j'*U_j and the positive eigenvectors of H_j;
% outside W the constraint is just H restricted there (negative definite),
% so it is eliminated exactly by a Schur complement.
n = size(H,1);
[Vh,Dh] = eig(H); Vpos = Vh(:, real(diag(Dh)) > 1e-10);
W = orth([U1, U2, A'*U1, Vpos]);
if size(W,2) >= n, W = eye(n); T = W; R = H; return; end
Q = null(W');
Hqq = Q'*H*Q; Hqq = (Hqq+Hqq')/2;
if max(real(eig(Hqq))) >= 0, error('Q''HQ not negative definite'); end
F = Hqq \ (Q'*H*W);
T = W - Q*F;                  % lift map: full multiplier = T*y*T'
R = T'*H*T; R = (R+R')/2;     % = W'HW - W'HQ (Q'HQ)^{-1} Q'HW
end

function Eb = hbasis(r)
% columns = vec of a real basis of r x r Hermitian matrices (r^2 of them)
Eb = zeros(r*r, r*r); k = 0;
for a = 1:r, k = k+1; E = zeros(r); E(a,a) = 1; Eb(:,k) = E(:); end
for a = 1:r
    for b = a+1:r
        k = k+1; E = zeros(r); E(a,b) = 1;  E(b,a) = 1;   Eb(:,k) = E(:);
        k = k+1; E = zeros(r); E(a,b) = 1i; E(b,a) = -1i; Eb(:,k) = E(:);
    end
end
end

function Gv = embvec(Mv, m)
% columns of Mv are vec of Hermitian m x m matrices; returns vec(emb(M)) columns
p = size(Mv,2); Gv = zeros(4*m*m, p);
for k = 1:p
    M = reshape(Mv(:,k), m, m); M = (M+M')/2;
    E = [real(M), -imag(M); imag(M), real(M)]; Gv(:,k) = E(:);
end
end

function Y = psdpart(Y)
% nearest PSD matrix (drop negative eigenvalues caused by solver round-off)
Y = (Y+Y')/2; [V,D] = eig(Y); d = max(real(diag(D)),0); Y = V*diag(d)*V'; Y = (Y+Y')/2;
end

function E = emb(X)
% real 2m x 2m form of a complex Hermitian m x m matrix (works on sdpvars too)
E = [real(X), -imag(X); imag(X), real(X)];
end

function Y = unemb(Z)
% complex Hermitian multiplier from the dual Z of an embedded constraint:
% trace(Z*emb(M)) = real(trace(Y*M)) for every Hermitian M.
if any(isnan(Z(:))), error('Solver returned no dual values (NaN). Check YALMIP/solver setup.'); end
m = size(Z,1)/2; Z = (Z+Z')/2;
Y = (Z(1:m,1:m) + Z(m+1:end,m+1:end)) + 1i*(Z(m+1:end,1:m) - Z(1:m,m+1:end));
Y = (Y+Y')/2; if real(trace(Y)) < 0, Y = -Y; end
end

function U = topvecs(X, r)
[V,D] = eig((X+X')/2); [~,ix] = sort(real(diag(D)),'descend'); U = V(:,ix(1:r));
end

function A = build_A(N, o)
% Identical to transcient_growth_new.m (finitediff dx = L/N, clamped D4,
% energy weight V, A = V*(-1i*L)/V, t = t0 + i*dt, i = 1..K).
L = 2; dx = L/N; Nd = N-2;
x = linspace(-L/2, L/2, N); x = x(2:end-1).';
D1 = (diag(-ones(1,Nd-1),1) + diag(ones(1,Nd-1),-1))/(2*dx);
D2 = (diag(-2*ones(1,Nd)) + diag(ones(1,Nd-1),1) + diag(ones(1,Nd-1),-1))/dx^2;
D4 = diag(6*ones(1,Nd)) + diag(-4*ones(1,Nd-1),1) + diag(-4*ones(1,Nd-1),-1) ...
     + diag(ones(1,Nd-2),2) + diag(ones(1,Nd-2),-2);
D4(1,1) = D4(1,1)+1; D4(end,end) = D4(end,end)+1; D4 = D4/dx^4;
kx = o.kx; kz = o.kz; Re = o.Re; kappa = o.kappa;
k2 = (kx^2+kz^2)*eye(Nd); kk = sqrt(kx^2+kz^2)*eye(Nd);
W8 = diag(sqrt(dx*ones(1,Nd)));
V = [W8*(D1+kk), zeros(Nd); zeros(Nd), W8];
c = D2 - k2;  oo4 = 1/(1i*Re)*(D4 - 2*D2*k2 + k2*k2);  oo = -inv(c);  ooo = (1/(1i*Re))*c;
K = round((o.T-o.t0)/o.dt); A = cell(K,1);
for i = 1:K
    t = o.t0 + i*o.dt;
    g = exp(-kappa*t); dg = -kappa*exp(-kappa*t);
    U = zeros(Nd,1); Uy = U; Uyy = U;
    for m = 1:o.NN
        an = (pi*m)^2/Re; d = an - kappa;
        q = exp(-min(kappa,an)*t)*(-expm1(-abs(d)*t))/abs(d);
        amp = -2*Re*(-1)^m/(pi*m)^3*(kappa^2*q - kappa*exp(-an*t));
        U = U + amp*sin(m*pi*x); Uy = Uy + amp*(m*pi)*cos(m*pi*x); Uyy = Uyy - amp*(m*pi)^2*sin(m*pi*x);
    end
    U = U + Re/6*dg*(x.^3 - x) + g*x; Uy = Uy + Re/6*dg*(3*x.^2-1) + g; Uyy = Uyy + Re/6*dg*(6*x);
    Los = oo*(oo4 - kx*diag(U)*c + kx*diag(Uyy));
    Lsq = kx*diag(U) - ooo;  Lc = kz*diag(Uy);
    A{i} = V*(-1i*[Los, zeros(Nd); Lc, Lsq])/V;
end
end
