%% PROJECT 17: Compliant Mechanism Continuum Inverse Kinematics
% Models a constant curvature soft robot arm
% Forward kinematics: arc parameters -> tip position
% Inverse kinematics: target position -> arc parameters via fmincon

%--- Forward Kinematics ---
function tip = forward_kinematics(params)
    kappa = params(1);
    l     = params(2);
    phi   = params(3);
    if abs(kappa) < 1e-6
        tip = [l*cos(phi); l*sin(phi); 0];
    else
        tip = [(1/kappa)*(1-cos(kappa*l))*cos(phi);
               (1/kappa)*(1-cos(kappa*l))*sin(phi);
                (1/kappa)*sin(kappa*l)];
    end
end

%--- Inverse Kinematics ---
function [params_opt, error] = inverse_kinematics(target, l_max, k_max)
    objective = @(p) norm(forward_kinematics(p) - target')^2;
    p0 = [5.0; l_max/2; atan2(target(2), target(1))];
    lb = [-k_max; 0.01; -pi];
    ub = [ k_max; l_max;  pi];
    options = optimoptions('fmincon', ...
        'Algorithm',           'sqp', ...
        'Display',             'off', ...
        'MaxIterations',       1000, ...
        'OptimalityTolerance', 1e-10);
    [params_opt, fval] = fmincon(objective, p0, ...
                                  [], [], [], [], lb, ub, [], options);
    error = sqrt(fval);
end

%--- Plot Arm ---
function plot_arm(params, rgb_color, label)
    kappa = params(1);
    l     = params(2);
    phi   = params(3);
    s     = linspace(0, l, 100);
    arm_pts = zeros(3, length(s));
    for i = 1:length(s)
        arm_pts(:,i) = forward_kinematics([kappa; s(i); phi]);
    end
    plot3(arm_pts(1,:), arm_pts(2,:), arm_pts(3,:), ...
          'Color', rgb_color, 'LineWidth', 3, 'DisplayName', label)
    tip = forward_kinematics(params);
    plot3(tip(1), tip(2), tip(3), 'o', 'MarkerSize', 10, ...
          'MarkerFaceColor', rgb_color, 'MarkerEdgeColor', 'k', ...
          'HandleVisibility', 'off')
end

%--- Robot Parameters ---
l_max = 0.3;
k_max = 20;

%--- Target Points ---
targets = [0.15,  0.10,  0.05;
           0.10, -0.10,  0.15;
           0.20,  0.05,  0.10];

fprintf('--- Continuum Robot Inverse Kinematics ---\n')
fprintf('Max arc length : %.2f m\n', l_max)
fprintf('Max curvature  : %.1f 1/m\n', k_max)
fprintf('Targets        : %d\n', size(targets,1))

%--- Solve IK ---
fprintf('\nSolving inverse kinematics...\n')
fprintf('Target | kappa  |   l    |  phi   | Error\n')
fprintf('-------|--------|--------|--------|------\n')

params_all = zeros(size(targets,1), 3);
for i = 1:size(targets,1)
    [params, err] = inverse_kinematics(targets(i,:), l_max, k_max);
    params_all(i,:) = params;
    fprintf('%6d | %6.2f | %6.4f | %6.3f | %.2e m\n', ...
             i, params(1), params(2), params(3), err)
end

%--- Plot ---
colors = {[0 0 1], [1 0 0], [0 0.7 0]};    % RGB: blue, red, green

figure;
hold on

for i = 1:size(targets,1)
    plot_arm(params_all(i,:), colors{i}, sprintf('Arm to Target %d', i))
    plot3(targets(i,1), targets(i,2), targets(i,3), ...
          'kx', 'MarkerSize', 15, 'LineWidth', 3, ...
          'HandleVisibility', 'off')
end

plot3(0, 0, 0, 'ks', 'MarkerSize', 15, 'MarkerFaceColor', 'k', ...
      'DisplayName', 'Base')

legend('Location', 'northwest')
title('Continuum Robot - Constant Curvature IK Solutions')
xlabel('x (m)')
ylabel('y (m)')
zlabel('z (m)')
grid on
axis equal
view(45, 30)
hold off

%--- Print results ---
fprintf('\n--- IK Results ---\n')
for i = 1:size(targets,1)
    tip = forward_kinematics(params_all(i,:));
    fprintf('Target %d: [%.3f, %.3f, %.3f]\n', ...
             i, targets(i,1), targets(i,2), targets(i,3))
    fprintf('  Tip   : [%.3f, %.3f, %.3f]\n', tip(1), tip(2), tip(3))
    fprintf('  kappa=%.2f, l=%.4f, phi=%.3f\n', ...
             params_all(i,1), params_all(i,2), params_all(i,3))
end

saveas(gcf, 'continuum_robot.png')