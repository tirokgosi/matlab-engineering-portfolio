%% PROJECT 7: Trajectory Optimization with Direct Collocation
% Finds minimum acceleration path for a 2D quadcopter
% avoiding a circular obstacle using fmincon SQP

%--- Problem Parameters ---
% Start and end points
A = [0, 0];         % Start point (x,y) in meters
B = [10, 0];        % End point (x,y) in meters

%--- Obstacle ---
obs_center = [5, 0];    % Obstacle center (right in the way)
obs_radius = 2.0;       % Obstacle radius in meters

%--- Time Discretization ---
N  = 20;            % Number of time nodes
T  = 5.0;           % Total flight time in seconds
dt = T/(N-1);       % Time step between nodes

%--- Decision Variable Layout ---
% At each node we have: x, y, vx, vy, ax, ay = 6 variables
% Total decision variables = 6 * N
n_vars = 6 * N;

%--- Print setup ---
fprintf('--- Trajectory Optimization Setup ---\n')
fprintf('Start point    : [%.1f, %.1f]\n', A(1), A(2))
fprintf('End point      : [%.1f, %.1f]\n', B(1), B(2))
fprintf('Obstacle center: [%.1f, %.1f]\n', obs_center(1), obs_center(2))
fprintf('Obstacle radius: %.1f m\n', obs_radius)
fprintf('Time nodes     : %d\n', N)
fprintf('Total time     : %.1f seconds\n', T)

%--- Objective Function ---
% Minimize sum of squared accelerations (minimum effort)
% Decision vector Z = [x1,y1,vx1,vy1,ax1,ay1, x2,y2,vx2,vy2,ax2,ay2, ...]
objective = @(Z) sum(Z(5:6:end).^2 + Z(6:6:end).^2);

%--- Extract State Variables from Decision Vector ---
% Helper function to unpack Z into readable arrays
function [x,y,vx,vy,ax,ay] = unpack(Z)
x  = Z(1:6:end);
y  = Z(2:6:end);
vx = Z(3:6:end);
vy = Z(4:6:end);
ax = Z(5:6:end);
ay = Z(6:6:end);
end

%--- Nonlinear Constraints ---
function [c, ceq] = constraints(Z, N, dt, A, B, obs_center, obs_radius)

[x,y,vx,vy,ax,ay] = unpack(Z);

%--- Equality Constraints (ceq = 0) ---
ceq = [];

% 1. Boundary conditions: start at A with zero velocity
ceq = [ceq; x(1)  - A(1)];     % Start x
ceq = [ceq; y(1)  - A(2)];     % Start y
ceq = [ceq; vx(1) - 0];        % Start vx = 0
ceq = [ceq; vy(1) - 0];        % Start vy = 0

% 2. Boundary conditions: end at B with zero velocity
ceq = [ceq; x(end)  - B(1)];   % End x
ceq = [ceq; y(end)  - B(2)];   % End y
ceq = [ceq; vx(end) - 0];      % End vx = 0
ceq = [ceq; vy(end) - 0];      % End vy = 0

% 3. Dynamics constraints: trapezoidal collocation
% Position update: x(i+1) = x(i) + dt/2*(vx(i) + vx(i+1))
% Velocity update: vx(i+1) = vx(i) + dt/2*(ax(i) + ax(i+1))
for i = 1:N-1
    ceq = [ceq; x(i+1)  - x(i)  - dt/2*(vx(i) + vx(i+1))];
    ceq = [ceq; y(i+1)  - y(i)  - dt/2*(vy(i) + vy(i+1))];
    ceq = [ceq; vx(i+1) - vx(i) - dt/2*(ax(i) + ax(i+1))];
    ceq = [ceq; vy(i+1) - vy(i) - dt/2*(ay(i) + ay(i+1))];
end

%--- Inequality Constraints (c <= 0) ---
% Path constraint: stay outside obstacle
% Distance to obstacle center must be >= radius
% Rewritten as: radius - distance <= 0
c = [];
for i = 1:N
    dist = sqrt((x(i)-obs_center(1))^2 + (y(i)-obs_center(2))^2);
    c = [c; obs_radius - dist];
end

end

%--- Initial Guess ---
% Straight line from A to B with zero velocity and acceleration
Z0 = zeros(6*N, 1);
for i = 1:N
    frac = (i-1)/(N-1);
    Z0(6*i-5) = A(1) + frac*(B(1)-A(1));   % x linearly interpolated
    Z0(6*i-4) = A(2) + frac*(B(2)-A(2));   % y linearly interpolated
end

%--- fmincon Options ---
options = optimoptions('fmincon', ...
    'Algorithm',            'sqp', ...
    'Display',              'iter', ...
    'MaxIterations',        500, ...
    'MaxFunctionEvaluations', 50000, ...
    'OptimalityTolerance',  1e-6, ...
    'ConstraintTolerance',  1e-6);

%--- Solve ---
fprintf('\nRunning fmincon SQP optimizer...\n')
con = @(Z) constraints(Z, N, dt, A, B, obs_center, obs_radius);
[Z_opt, fval, exitflag] = fmincon(objective, Z0, [], [], [], [], ...
    [], [], con, options);

fprintf('\nOptimization complete\n')
fprintf('Exit flag = %d (1 = success)\n', exitflag)
fprintf('Minimum control effort = %.4f\n', fval)

%--- Unpack Solution ---
[x,y,vx,vy,ax,ay] = unpack(Z_opt);

%--- Plot ---
figure;
hold on;

% Draw obstacle
theta_obs = linspace(0, 2*pi, 100);
obs_x = obs_center(1) + obs_radius*cos(theta_obs);
obs_y = obs_center(2) + obs_radius*sin(theta_obs);
fill(obs_x, obs_y, 'r', 'FaceAlpha', 0.4, 'EdgeColor', 'r')
text(obs_center(1), obs_center(2), 'OBSTACLE', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold')

% Draw straight line path (naive solution)
plot([A(1), B(1)], [A(2), B(2)], 'k--', 'LineWidth', 1.5)

% Draw optimal path
plot(x, y, 'b-o', 'LineWidth', 2.5, 'MarkerSize', 6)

% Draw thrust vectors
quiver(x, y, ax, ay, 0.3, 'g', 'LineWidth', 1.5)

% Mark start and end
plot(A(1), A(2), 'gs', 'MarkerSize', 15, 'MarkerFaceColor', 'g')
plot(B(1), B(2), 'rs', 'MarkerSize', 15, 'MarkerFaceColor', 'r')
text(A(1)-0.5, A(2)+0.3, 'START', 'FontWeight', 'bold', 'Color', 'g')
text(B(1)-0.5, B(2)+0.3, 'END',   'FontWeight', 'bold', 'Color', 'r')

legend('Obstacle', 'Straight line', 'Optimal path', 'Thrust vectors')
title('Quadcopter Trajectory Optimization - Direct Collocation')
xlabel('x (m)')
ylabel('y (m)')
axis equal
grid on
hold off

%--- Save ---
saveas(gcf, 'trajectory_optimization.png')