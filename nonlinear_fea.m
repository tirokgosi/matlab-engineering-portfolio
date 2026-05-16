%% PROJECT 21: Nonlinear FEA with Newton-Raphson (Geometric Nonlinearity)
% 2D truss with large displacements
% Green-Lagrange strain, 2nd Piola-Kirchhoff stress
% Newton-Raphson iteration at each load increment
% Compares nonlinear vs linear load-deflection curve

%--- Truss Definition ---
nodes    = [0, 0; 1, 1; 2, 0];
elements = [1,2; 2,3; 1,3];
E = 200e9;
A = 0.01;
n_nodes   = size(nodes,1);
n_elements = size(elements,1);
ndof      = 2*n_nodes;

%--- Load ---
F_total      = zeros(ndof,1);
F_total(4)   = -200000;     % 200 kN downward at Node 2
n_increments = 20;
tol          = 1e-6;
max_iter     = 50;

%--- Storage ---
U_hist      = zeros(ndof, n_increments);
load_hist   = zeros(1,    n_increments);
U_lin_hist  = zeros(ndof, n_increments);

fprintf('--- Nonlinear FEA with Newton-Raphson ---\n')
fprintf('Elements    : %d\n', n_elements)
fprintf('DOFs        : %d\n', ndof)
fprintf('Load increments: %d\n', n_increments)
fprintf('Tolerance   : %.2e\n', tol)

%--- Linear Stiffness Matrix ---
function K = assemble_linear(nodes, elements, E, A)
    n  = 2*size(nodes,1);
    K  = zeros(n,n);
    for i = 1:size(elements,1)
        n1 = elements(i,1); n2 = elements(i,2);
        x1 = nodes(n1,1); y1 = nodes(n1,2);
        x2 = nodes(n2,1); y2 = nodes(n2,2);
        L  = sqrt((x2-x1)^2+(y2-y1)^2);
        c  = (x2-x1)/L; s = (y2-y1)/L;
        k  = E*A/L*[c*c,c*s,-c*c,-c*s;
                    c*s,s*s,-c*s,-s*s;
                   -c*c,-c*s,c*c,c*s;
                   -c*s,-s*s,c*s,s*s];
        dofs = [2*n1-1,2*n1,2*n2-1,2*n2];
        K(dofs,dofs) = K(dofs,dofs)+k;
    end
end

%--- Nonlinear Internal Force and Tangent Stiffness ---
function [F_int, K_t] = nonlinear_element(nodes, elements, U, E, A)
    n     = 2*size(nodes,1);
    F_int = zeros(n,1);
    K_t   = zeros(n,n);
    for i = 1:size(elements,1)
        n1 = elements(i,1); n2 = elements(i,2);
        x1 = nodes(n1,1); y1 = nodes(n1,2);
        x2 = nodes(n2,1); y2 = nodes(n2,2);
        L0 = sqrt((x2-x1)^2+(y2-y1)^2);
        dofs = [2*n1-1,2*n1,2*n2-1,2*n2];
        u_e  = U(dofs);

        % Deformed coordinates
        dx = (x2+u_e(3)) - (x1+u_e(1));
        dy = (y2+u_e(4)) - (y1+u_e(2));
        L  = sqrt(dx^2+dy^2);

        % Green-Lagrange strain
        E_gl = (L^2 - L0^2)/(2*L0^2);

        % 2nd Piola-Kirchhoff stress
        S    = E*E_gl;

        % Force in deformed direction
        F_e  = E*A/L0 * E_gl * L/L0;
        c    = dx/L; s = dy/L;
        f_e  = F_e*[-c;-s;c;s];
        F_int(dofs) = F_int(dofs) + f_e;

        % Tangent stiffness
        k_mat = E*A*E_gl/L0 * ...
                [1,0,-1,0;0,1,0,-1;-1,0,1,0;0,-1,0,1];
        k_geo = E*A/(L0^3)*[dx^2,dx*dy,-dx^2,-dx*dy;
                             dx*dy,dy^2,-dx*dy,-dy^2;
                            -dx^2,-dx*dy,dx^2,dx*dy;
                            -dx*dy,-dy^2,dx*dy,dy^2];
        K_t(dofs,dofs) = K_t(dofs,dofs) + k_mat + k_geo;
    end
end

%--- Boundary Conditions ---
fixed = [1,2,6];
free  = setdiff(1:ndof, fixed);
penalty = 1e20;

%--- Nonlinear Solution ---
U = zeros(ndof,1);
fprintf('\nIter | Load%% | Residual\n')
fprintf('-----|-------|----------\n')

for inc = 1:n_increments
    lambda   = inc/n_increments;
    F_ext    = lambda * F_total;

    for iter = 1:max_iter
        [F_int, K_t] = nonlinear_element(nodes, elements, U, E, A);
        K_t(1,1)     = K_t(1,1)+penalty;
        K_t(2,2)     = K_t(2,2)+penalty;
        K_t(6,6)     = K_t(6,6)+penalty;
        R            = F_ext - F_int;
        R([1,2,6])   = 0;
        res          = norm(R);
        if res < tol; break; end
        dU           = K_t\R;
        U            = U + dU;
    end

    U_hist(:,inc)    = U;
    load_hist(inc)   = lambda;
    fprintf('%4d | %5.1f%% | %.4e\n', inc, lambda*100, res)
end

%--- Linear Solution for Comparison ---
K_lin = assemble_linear(nodes, elements, E, A);
K_lin(1,1) = K_lin(1,1)+penalty;
K_lin(2,2) = K_lin(2,2)+penalty;
K_lin(6,6) = K_lin(6,6)+penalty;

for inc = 1:n_increments
    lambda         = inc/n_increments;
    U_lin          = K_lin \ (lambda*F_total);
    U_lin_hist(:,inc) = U_lin;
end

fprintf('\nNonlinear analysis complete\n')
fprintf('Final displacement Node2 y: %.6f m\n', U_hist(4,end))
fprintf('Linear    displacement     : %.6f m\n', U_lin_hist(4,end))

%--- Plot ---
figure;

subplot(1,2,1)
plot(abs(U_hist(4,:))*1000,    load_hist*abs(F_total(4))/1000, ...
     'b-o', 'LineWidth', 2, 'MarkerSize', 5)
hold on
plot(abs(U_lin_hist(4,:))*1000, load_hist*abs(F_total(4))/1000, ...
     'r--o', 'LineWidth', 2, 'MarkerSize', 5)
legend({'Nonlinear (NR)','Linear'}, 'Location', 'best')
title('Load-Deflection Curve')
xlabel('Vertical Displacement (mm)')
ylabel('Applied Load (kN)')
grid on
hold off

subplot(1,2,2)
scale   = 5;
nodes_d = nodes + scale*[U_hist(1:2:end,end), U_hist(2:2:end,end)];
for i = 1:size(elements,1)
    n1 = elements(i,1); n2 = elements(i,2);
    plot([nodes(n1,1),nodes(n2,1)],[nodes(n1,2),nodes(n2,2)], ...
         'b--', 'LineWidth', 1.5)
    hold on
    plot([nodes_d(n1,1),nodes_d(n2,1)],[nodes_d(n1,2),nodes_d(n2,2)], ...
         'r-', 'LineWidth', 2.5)
end
legend({'Undeformed','Deformed (5x)'}, 'Location', 'best')
title('Deformed Shape (5x scaled)')
xlabel('x (m)'); ylabel('y (m)')
axis equal; grid on; hold off

sgtitle('Nonlinear FEA - Newton-Raphson Iteration')

saveas(gcf, 'nonlinear_fea.png')