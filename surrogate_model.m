%% PROJECT 14: Machine Learning Surrogate Model for FEA
% Trains a Gaussian Process Regression model on truss FEA data
% Replaces expensive FEA with instant surrogate predictions
% Shows predicted stress surface with confidence intervals

%--- Material Properties ---
E = 200e9;      % Young's modulus in Pa
A = 0.01;       % Cross sectional area in m^2

%--- Design Space ---
% Node 2 position varies in x and y
% Node 1 fixed at [0,0], Node 3 fixed at [2,0]
x2_range = linspace(0.5, 1.5, 10);     % Node 2 x position
y2_range = linspace(0.5, 1.5, 10);     % Node 2 y position

[X2, Y2] = meshgrid(x2_range, y2_range);
n_samples = numel(X2);

%--- Storage ---
max_stress = zeros(n_samples, 1);

fprintf('--- Surrogate Model for Truss FEA ---\n')
fprintf('Training samples  : %d\n', n_samples)
fprintf('Design variables  : Node 2 x and y position\n')
fprintf('\nGenerating FEA training data...\n')

%--- Run Truss Solver for Each Design ---
for s = 1:n_samples

    % Node 2 position for this sample
    x2 = X2(s);
    y2 = Y2(s);

    % Node coordinates
    nodes = [0,  0;
        x2, y2;
        2,  0];

    % Element connectivity
    elements = [1,2; 2,3; 1,3];

    % Load vector - downward force at Node 2
    F = zeros(6,1);
    F(4) = -50000;

    % Assemble global stiffness matrix
    K_global = zeros(6,6);
    for i = 1:3
        n1 = elements(i,1);
        n2 = elements(i,2);
        x1 = nodes(n1,1); y1 = nodes(n1,2);
        x2e = nodes(n2,1); y2e = nodes(n2,2);
        L  = sqrt((x2e-x1)^2 + (y2e-y1)^2);
        c  = (x2e-x1)/L;
        s_ang = (y2e-y1)/L;
        k  = (E*A/L)*[c*c, c*s_ang, -c*c, -c*s_ang;
            c*s_ang, s_ang^2, -c*s_ang, -s_ang^2;
            -c*c, -c*s_ang, c*c, c*s_ang;
            -c*s_ang, -s_ang^2, c*s_ang, s_ang^2];
        dofs = [2*n1-1, 2*n1, 2*n2-1, 2*n2];
        K_global(dofs,dofs) = K_global(dofs,dofs) + k;
    end

    % Boundary conditions
    penalty = 1e20;
    K_global(1,1) = K_global(1,1) + penalty;
    K_global(2,2) = K_global(2,2) + penalty;
    K_global(6,6) = K_global(6,6) + penalty;

    % Solve
    U = K_global \ F;

    % Calculate stress in each element
    stresses = zeros(3,1);
    for i = 1:3
        n1  = elements(i,1);
        n2e = elements(i,2);
        x1  = nodes(n1,1);  y1 = nodes(n1,2);
        x2c = nodes(n2e,1); y2c = nodes(n2e,2);
        L   = sqrt((x2c-x1)^2+(y2c-y1)^2);
        c   = (x2c-x1)/L;
        sc  = (y2c-y1)/L;
        dofs = [2*n1-1,2*n1,2*n2e-1,2*n2e];
        u_e  = U(dofs);
        stresses(i) = E/L*[-c,-sc,c,sc]*u_e;
    end

    max_stress(s) = max(abs(stresses));

end

fprintf('FEA data generation complete\n')
fprintf('Stress range: [%.2e, %.2e] Pa\n', ...
    min(max_stress), max(max_stress))
%--- Prepare Training Data ---
% Inputs: Node 2 x and y positions
% Output: Maximum stress
X_train = [X2(:), Y2(:)];      % 100 x 2 input matrix
y_train = max_stress;           % 100 x 1 output vector

%--- Train Gaussian Process Regression Model ---
fprintf('\nTraining Gaussian Process Regression model...\n')

gpr_model = fitrgp(X_train, y_train, ...
    'KernelFunction',       'squaredexponential', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('ShowPlots', false, 'Verbose', 0));

fprintf('GPR model trained successfully\n')

%--- Evaluate Model on Fine Grid ---
x2_fine = linspace(0.5, 1.5, 50);
y2_fine = linspace(0.5, 1.5, 50);
[X2_fine, Y2_fine] = meshgrid(x2_fine, y2_fine);
X_test  = [X2_fine(:), Y2_fine(:)];

%--- Predict with Confidence Intervals ---
[y_pred, ~, y_ci] = predict(gpr_model, X_test);

y_pred_grid = reshape(y_pred, 50, 50);
y_ci_grid   = reshape(y_ci(:,2) - y_ci(:,1), 50, 50);

%--- Model Performance ---
[y_train_pred, ~] = predict(gpr_model, X_train);
rmse = sqrt(mean((y_train_pred - y_train).^2));
r2   = 1 - sum((y_train - y_train_pred).^2) / ...
    sum((y_train - mean(y_train)).^2);

fprintf('\n--- GPR Model Performance ---\n')
fprintf('Training RMSE : %.4e Pa\n', rmse)
fprintf('R-squared     : %.6f\n', r2)
fprintf('(R2 = 1.0 is perfect fit)\n')
%--- Plot Results ---
figure;

%--- 3D Predicted Stress Surface ---
subplot(1,2,1)
surf(X2_fine, Y2_fine, y_pred_grid/1e6, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.8)
hold on

% Overlay true FEA training points
scatter3(X2(:), Y2(:), max_stress/1e6, ...
    50, 'r', 'filled', 'DisplayName', 'FEA training points')

colorbar
colormap('jet')
xlabel('Node 2 x position (m)')
ylabel('Node 2 y position (m)')
zlabel('Max Stress (MPa)')
title('GPR Surrogate - Predicted Stress Surface')
legend('GPR prediction', 'FEA training points')
view(45, 30)
grid on
hold off

%--- Confidence Interval Plot ---
subplot(1,2,2)
contourf(X2_fine, Y2_fine, y_ci_grid/1e6, 20, 'LineStyle', 'none')
colorbar
hold on
scatter(X2(:), Y2(:), 30, 'w', 'filled')
xlabel('Node 2 x position (m)')
ylabel('Node 2 y position (m)')
title('GPR Prediction Uncertainty (95% CI Width)')
colormap('hot')
grid on
hold off

%--- Print key findings ---
[min_stress, idx_min] = min(y_pred);
[max_stress_pred, idx_max] = max(y_pred);

fprintf('\n--- Surrogate Predictions ---\n')
fprintf('Minimum predicted stress: %.2f MPa\n', min_stress/1e6)
fprintf('  at Node2 = [%.2f, %.2f] m\n', ...
    X_test(idx_min,1), X_test(idx_min,2))
fprintf('Maximum predicted stress: %.2f MPa\n', max_stress_pred/1e6)
fprintf('  at Node2 = [%.2f, %.2f] m\n', ...
    X_test(idx_max,1), X_test(idx_max,2))
fprintf('\nSurrogate replaces FEA in milliseconds vs seconds per run\n')

%--- Save ---
saveas(gcf, 'surrogate_model.png')