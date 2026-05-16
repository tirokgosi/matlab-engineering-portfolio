%% PROJECT 25: Isogeometric Analysis of a Curved Beam
% Uses NURBS basis functions directly from CAD geometry
% No meshing approximation - exact geometry representation
% Solves for displacement of a curved cantilever beam

%--- Beam Properties ---
E    = 200e9;       % Young's modulus Pa
nu   = 0.3;         % Poisson's ratio
t    = 0.01;        % Beam thickness m
b    = 0.05;        % Beam width m
A    = b*t;         % Cross sectional area m^2
I    = b*t^3/12;    % Second moment of area m^4
EI   = E*I;         % Bending stiffness
EA   = E*A;         % Axial stiffness

%--- NURBS Curve Definition ---
% Quadratic NURBS for a quarter circle arc
% Control points for quarter circle of radius R
R    = 1.0;         % Radius of curvature m
P    = 1000;        % Applied tip load N

% Control points [x, y, w] - w is NURBS weight
CP   = [R,   0,   1;
        R,   R,   1/sqrt(2);
        0,   R,   1];

n_cp = size(CP,1);  % Number of control points

% Knot vector for quadratic (p=2) NURBS
p    = 2;           % Polynomial degree
Xi   = [0,0,0,1,1,1];  % Open knot vector

fprintf('--- Isogeometric Analysis Curved Beam ---\n')
fprintf('Beam properties:\n')
fprintf('  E  = %.0f GPa\n', E/1e9)
fprintf('  R  = %.1f m\n', R)
fprintf('  EI = %.4f Nm^2\n', EI)
fprintf('  EA = %.0f N\n', EA)
fprintf('NURBS degree    : %d\n', p)
fprintf('Control points  : %d\n', n_cp)
fprintf('Applied load    : %.0f N\n', P)

%--- NURBS Basis Functions ---
function N = nurbs_basis(xi, Xi, p, w)
    % Compute NURBS basis functions at parameter xi
    n   = length(Xi)-p-2;
    B   = zeros(1,n+1);

    % B-spline basis via Cox-de Boor
    for i = 1:n+1
        B(i) = cox_deboor(xi, i, p, Xi);
    end

    % NURBS weights
    W   = sum(B.*w);
    N   = B.*w/W;
end

function N = cox_deboor(xi, i, p, Xi)
    if p == 0
        if Xi(i) <= xi && xi < Xi(i+1)
            N = 1;
        elseif xi == Xi(end) && Xi(i) < Xi(i+1)
            N = 1;
        else
            N = 0;
        end
        return
    end

    N = 0;
    d1 = Xi(i+p)   - Xi(i);
    d2 = Xi(i+p+1) - Xi(i+1);

    if d1 > 1e-10
        N = N + (xi-Xi(i))/d1 * cox_deboor(xi,i,p-1,Xi);
    end
    if d2 > 1e-10
        N = N + (Xi(i+p+1)-xi)/d2 * cox_deboor(xi,i+1,p-1,Xi);
    end
end

%--- Evaluate NURBS Curve ---
n_pts  = 100;
xi_pts = linspace(0,1,n_pts);
curve  = zeros(n_pts,2);

w_cp   = CP(:,3);

for k = 1:n_pts
    xi   = xi_pts(k);
    % Avoid exactly 1.0 for open knot vector
    xi   = min(xi, 1-1e-10);
    N    = nurbs_basis(xi, Xi, p, w_cp');
    curve(k,:) = N * CP(:,1:2);
end

%--- Analytical Solution for Curved Beam ---
% Tip deflection of curved cantilever under tip load
% delta = P*R^3/(EI) * (pi/4 - 2/pi) for quarter circle
delta_tip_v = P*R^3/EI * (pi/4);      % Vertical deflection
delta_tip_h = P*R^3/EI * (1 - pi/4);  % Horizontal deflection

fprintf('\n--- Analytical Solution ---\n')
fprintf('Tip vertical deflection   : %.6f m\n', delta_tip_v)
fprintf('Tip horizontal deflection : %.6f m\n', delta_tip_h)
fprintf('Max deflection/R ratio    : %.4f\n', delta_tip_v/R)

%--- IGA Stiffness Assembly (simplified Euler-Bernoulli) ---
% Use Gauss quadrature along the NURBS parameter
n_gauss = 10;
[xi_g, w_g] = gauss_points(n_gauss);

ndof   = 2*n_cp;
K_iga  = zeros(ndof, ndof);
F_iga  = zeros(ndof, 1);

function [xg, wg] = gauss_points(n)
    % Gauss-Legendre points on [0,1]
    switch n
        case 3
            xg = [0.1127, 0.5, 0.8873];
            wg = [0.2778, 0.4444, 0.2778];
        case 5
            xg = [0.0469,0.2307,0.5,0.7693,0.9531];
            wg = [0.1185,0.2393,0.2844,0.2393,0.1185];
        otherwise
            xg = linspace(0.01,0.99,n);
            wg = ones(1,n)/n;
    end
end

% Assemble stiffness using numerical integration
for g = 1:n_gauss
    xi  = xi_g(g);
    wgt = w_g(g);

    % NURBS basis at this point
    N   = nurbs_basis(xi, Xi, p, w_cp');

    % Physical coordinates
    xy  = N * CP(:,1:2);

    % Arc length derivative (Jacobian)
    dxi = 0.001;
    N_f = nurbs_basis(min(xi+dxi,0.999), Xi, p, w_cp');
    N_b = nurbs_basis(max(xi-dxi,0.001), Xi, p, w_cp');
    dxy = (N_f - N_b) * CP(:,1:2) / (2*dxi);
    J   = norm(dxy);

    % Assemble mass-like matrix (simplified)
    for i = 1:n_cp
        for j = 1:n_cp
            K_iga(2*i-1,2*j-1) = K_iga(2*i-1,2*j-1) + ...
                                   EA*N(i)*N(j)*wgt*J;
            K_iga(2*i,  2*j)   = K_iga(2*i,  2*j)   + ...
                                   EI*N(i)*N(j)*wgt*J;
        end
    end
end

% Apply tip load at free end (last control point)
F_iga(end) = -P;

% Apply BC: fix first control point
free_dofs = 3:ndof;
K_red     = K_iga(free_dofs, free_dofs);
F_red     = F_iga(free_dofs);
U_red     = K_red \ F_red;

U_full    = zeros(ndof,1);
U_full(free_dofs) = U_red;

% Deformed control points
CP_def    = CP(:,1:2);
for i = 1:n_cp
    CP_def(i,1) = CP(i,1) + U_full(2*i-1);
    CP_def(i,2) = CP(i,2) + U_full(2*i);
end

% Deformed curve
curve_def = zeros(n_pts,2);
for k = 1:n_pts
    xi  = min(xi_pts(k), 1-1e-10);
    N   = nurbs_basis(xi, Xi, p, w_cp');
    curve_def(k,:) = N * CP_def;
end

fprintf('\n--- IGA Results ---\n')
fprintf('IGA tip displacement x: %.6f m\n', U_full(end-1))
fprintf('IGA tip displacement y: %.6f m\n', U_full(end))

%--- Plot ---
figure;

subplot(1,2,1)
plot(curve(:,1), curve(:,2), 'b-', 'LineWidth', 3)
hold on
plot(curve_def(:,1), curve_def(:,2), 'r-', 'LineWidth', 3)
plot(CP(:,1), CP(:,2), 'ko--', 'LineWidth', 1.5, 'MarkerSize', 8)
plot(CP_def(:,1), CP_def(:,2), 'rs--', 'LineWidth', 1.5, 'MarkerSize', 8)
legend({'Original NURBS beam','Deformed beam', ...
        'Control polygon','Deformed CP'}, 'Location', 'best')
title('IGA Curved Beam - Exact NURBS Geometry')
xlabel('x (m)'); ylabel('y (m)')
axis equal; grid on; hold off

subplot(1,2,2)
% Plot NURBS basis functions
xi_plot = linspace(0.001, 0.999, 200);
N_plot  = zeros(3, 200);
for k = 1:200
    N_plot(:,k) = nurbs_basis(xi_plot(k), Xi, p, w_cp');
end
plot(xi_plot, N_plot(1,:), 'b-', 'LineWidth', 2)
hold on
plot(xi_plot, N_plot(2,:), 'r-', 'LineWidth', 2)
plot(xi_plot, N_plot(3,:), 'g-', 'LineWidth', 2)
legend({'N_1','N_2','N_3'}, 'Location', 'best')
title('NURBS Basis Functions')
xlabel('Parameter \xi')
ylabel('N_i(\xi)')
grid on; hold off

sgtitle('Isogeometric Analysis - Quarter Circle Curved Beam')

saveas(gcf, 'iga_curved_beam.png')