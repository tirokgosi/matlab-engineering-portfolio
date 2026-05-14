%% PROJECT 8: Topology Optimization of a Cantilever Beam
% SIMP method - Solid Isotropic Material with Penalization
% Finds optimal material distribution for minimum compliance
% Fixed left edge, point load at bottom right corner

%--- Problem Parameters ---
nelx = 60;        % Number of elements in x direction
nely = 30;        % Number of elements in y direction
volfrac = 0.4;    % Volume fraction - use only 40% of material
penal = 3.0;      % Penalization factor
rmin = 1.5;       % Filter radius

%--- Material Properties ---
E0   = 1.0;       % Young's modulus of solid
Emin = 1e-9;      % Young's modulus of void
nu   = 0.3;       % Poisson's ratio

fprintf('--- Topology Optimization Setup ---\n')
fprintf('Mesh           : %d x %d elements\n', nelx, nely)
fprintf('Volume fraction: %.0f%%\n', volfrac*100)

%--- Element Stiffness Matrix ---
k  = [1/2-nu/6,   1/8+nu/8,  -1/4-nu/12, -1/8+3*nu/8, ...
     -1/4+nu/12, -1/8-nu/8,   nu/6,       1/8-3*nu/8];
KE = E0/(1-nu^2)*[ k(1), k(2), k(3), k(4), k(5), k(6), k(7), k(8);
                   k(2), k(1), k(8), k(7), k(6), k(5), k(4), k(3);
                   k(3), k(8), k(1), k(6), k(7), k(4), k(5), k(2);
                   k(4), k(7), k(6), k(1), k(8), k(3), k(2), k(5);
                   k(5), k(6), k(7), k(8), k(1), k(2), k(3), k(4);
                   k(6), k(5), k(4), k(3), k(2), k(1), k(8), k(7);
                   k(7), k(4), k(5), k(2), k(3), k(8), k(1), k(6);
                   k(8), k(3), k(2), k(5), k(4), k(7), k(6), k(1)];

%--- FE Indexing ---
nodenrs = reshape(1:(1+nelx)*(1+nely), 1+nely, 1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1, nelx*nely, 1);
edofMat = repmat(edofVec,1,8) + ...
          repmat([0,1,2*nely+[2,3,0,1],-2,-1], nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);

%--- Filter ---
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k  = 0;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
            for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                e2 = (i2-1)*nely+j2;
                k  = k+1;
                iH(k) = e1;
                jH(k) = e2;
                sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H  = sparse(iH,jH,sH);
Hs = sum(H,2);

%--- Boundary Conditions and Load ---
ndof      = 2*(nelx+1)*(nely+1);
F         = sparse(ndof,1);
U         = zeros(ndof,1);
F(ndof,1) = -1;                        % Load at bottom right corner
fixeddofs = [1:2*(nely+1)];            % Fix entire left edge
freedofs  = setdiff(1:ndof,fixeddofs);

%--- Initialize ---
x      = repmat(volfrac,nely,nelx);
xPhys  = x;
loop   = 0;
change = 1;

figure;
fprintf('\nIter | Compliance | Volume | Change\n')
fprintf('-----|------------|--------|-------\n')

%--- Main Loop ---
while change > 0.01 && loop < 200

    loop = loop+1;

    %--- FEA ---
    sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),64*nelx*nely,1);
    K  = sparse(iK,jK,sK);
    K  = (K+K')/2;
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);

    %--- Sensitivity ---
    ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2),nely,nelx);
    c  = sum(sum((Emin+xPhys.^penal*(E0-Emin)).*ce));
    dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
    dv = ones(nely,nelx);

    %--- Filter ---
    dc(:) = H*(dc(:)./Hs);
    dv(:) = H*(dv(:)./Hs);

    %--- Optimality Criteria ---
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move, ...
               x.*sqrt(max(0,-dc./dv/lmid))))));
        xPhys(:) = (H*xnew(:))./Hs;
        if sum(xPhys(:)) > volfrac*nelx*nely
            l1 = lmid;
        else
            l2 = lmid;
        end
    end

    change = max(abs(xnew(:)-x(:)));
    x      = xnew;

    %--- Plot ---
    colormap(gray);
    imagesc(1-real(xPhys));
    axis equal off tight;
    title(sprintf('Iter: %d | Compliance: %.4f | Vol: %.3f', ...
                   loop, c, mean(xPhys(:))))
    drawnow;

    fprintf('%4d | %10.4f | %6.3f | %6.4f\n', ...
             loop, c, mean(xPhys(:)), change)

end

fprintf('\nOptimization complete in %d iterations\n', loop)
saveas(gcf,'topology_optimization.png')