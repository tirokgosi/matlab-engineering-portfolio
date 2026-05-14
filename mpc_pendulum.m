%% PROJECT 10: Model Predictive Control for an Inverted Pendulum
% Stabilizes a cart-pole system with track and force constraints
% MPC optimizes over a future horizon at every time step

%--- System Parameters ---
M   = 1.0;      % Cart mass in kg
m   = 0.1;      % Pendulum mass in kg
L   = 0.5;      % Pendulum length in meters
g   = 9.81;     % Gravity in m/s^2
d   = 0.1;      % Cart damping

%--- Linearized State Space ---
% State: [x, x_dot, theta, theta_dot]
Jt = (M+m);
A  = [0,        1,            0,       0;
      0,    -d/Jt,      m*g/Jt,       0;
      0,        0,            0,       1;
      0, -d/(Jt*L), (M+m)*g/(Jt*L),  0];
B  = [0; 1/Jt; 0; 1/(Jt*L)];
C  = eye(4);
D  = zeros(4,1);

%--- Discretize ---
dt    = 0.05;
sys_c = ss(A, B, C, D);
sys_d = c2d(sys_c, dt, 'zoh');
Ad    = sys_d.A;
Bd    = sys_d.B;

fprintf('--- MPC Inverted Pendulum ---\n')
fprintf('System discretized successfully\n')

%--- MPC Parameters ---
Np  = 20;       % Prediction horizon
Nc  = 10;       % Control horizon
nx  = 4;        % Number of states
nu  = 1;        % Number of inputs

%--- Cost Matrices ---
Q = diag([10, 1, 100, 1]);
R = 0.01;

%--- Constraints ---
x_max =  2.0;
x_min = -2.0;
u_max =  20;
u_min = -20;

%--- Build Prediction Matrices ---
Phi   = zeros(nx*Np, nx);
Gamma = zeros(nx*Np, Nc);

for i = 1:Np
    Phi((i-1)*nx+1:i*nx, :) = Ad^i;
    for j = 1:min(i,Nc)
        Gamma((i-1)*nx+1:i*nx, j) = Ad^(i-j) * Bd;
    end
end

%--- QP Cost Matrices ---
Qbar = kron(eye(Np), Q);
Rbar = kron(eye(Nc), R);
H_qp = Gamma'*Qbar*Gamma + Rbar;
H_qp = (H_qp + H_qp')/2;

fprintf('\n--- MPC Parameters ---\n')
fprintf('Prediction horizon: %d steps (%.2f seconds)\n', Np, Np*dt)
fprintf('Control horizon   : %d steps\n', Nc)
fprintf('Track limits      : [%.1f, %.1f] meters\n', x_min, x_max)
fprintf('Force limits      : [%.1f, %.1f] Newtons\n', u_min, u_max)

%--- Simulation Setup ---
T_sim  = 10;
t      = 0:dt:T_sim;
Nsim   = length(t);
x0     = [0; 0; 0.2; 0];

X      = zeros(nx, Nsim);
U      = zeros(nu, Nsim-1);
X(:,1) = x0;

%--- fmincon Options ---
options = optimoptions('fmincon', ...
    'Algorithm',     'sqp', ...
    'Display',       'off', ...
    'MaxIterations', 200);

%--- Input Constraints ---
lb = u_min * ones(Nc,1);
ub = u_max * ones(Nc,1);

%--- State Constraints on Cart Position ---
A_con = zeros(2*Np, Nc);
b_con_base = zeros(2*Np, 1);
Phi_x  = zeros(Np, nx);
Gam_x  = zeros(Np, Nc);

for i = 1:Np
    row_x       = (i-1)*nx + 1;
    Phi_x(i,:)  = Phi(row_x,:);
    Gam_x(i,:)  = Gamma(row_x,:);
end

A_con(1:Np,    :) =  Gam_x;
A_con(Np+1:end,:) = -Gam_x;

fprintf('\nRunning MPC simulation...\n')

%--- MPC Loop ---
U_prev = zeros(Nc,1);

for k = 1:Nsim-1

    xk   = X(:,k);
    f_qp = Gamma'*Qbar*Phi*xk;

    %--- State constraint RHS ---
    b_con        = zeros(2*Np,1);
    b_con(1:Np)         =  x_max - Phi_x*xk;
    b_con(Np+1:end)     = -x_min + Phi_x*xk;

    %--- Solve QP ---
    [U_opt, ~, exitflag] = fmincon(...
        @(u) 0.5*u'*H_qp*u + f_qp'*u, ...
        U_prev, A_con, b_con, [], [], lb, ub, [], options);

    if exitflag < 0
        U_opt = U_prev;
    end

    %--- Apply first control ---
    U(k)   = U_opt(1);
    U_prev = [U_opt(2:end); U_opt(end)];

    %--- Simulate one step ---
    X(:,k+1) = Ad*xk + Bd*U(k);

    if mod(k,50) == 0
        fprintf('Step %3d | x=%.2fm | theta=%.3frad | u=%.2fN\n', ...
                 k, xk(1), xk(3), U(k))
    end

end

fprintf('MPC simulation complete\n')

%--- Plot ---
figure;

subplot(3,1,1)
plot(t, X(1,:), 'b-', 'LineWidth', 2)
hold on
yline( x_max, 'r--', 'Track Limit', 'LineWidth', 1.5)
yline( x_min, 'r--', 'LineWidth', 1.5)
legend('Cart Position', 'Track Limits')
title('MPC Inverted Pendulum - Cart Position')
xlabel('Time (seconds)')
ylabel('Position (m)')
ylim([x_min-0.5, x_max+0.5])
grid on
hold off

subplot(3,1,2)
plot(t, rad2deg(X(3,:)), 'r-', 'LineWidth', 2)
hold on
yline(0,   'k--', 'Upright',  'LineWidth', 1.5)
yline( 10, 'g--', '±10 deg', 'LineWidth', 1)
yline(-10, 'g--', 'LineWidth', 1)
legend('Pendulum Angle', 'Upright')
title('Pendulum Angle')
xlabel('Time (seconds)')
ylabel('Angle (degrees)')
grid on
hold off

subplot(3,1,3)
stairs(t(1:end-1), U, 'm-', 'LineWidth', 2)
hold on
yline( u_max, 'r--', 'Force Limit', 'LineWidth', 1.5)
yline( u_min, 'r--', 'LineWidth', 1.5)
legend('Control Force', 'Force Limits')
title('MPC Control Force')
xlabel('Time (seconds)')
ylabel('Force (N)')
ylim([u_min-2, u_max+2])
grid on
hold off

%--- Performance ---
fprintf('\n--- Performance ---\n')
fprintf('Initial angle     : %.1f degrees\n',  rad2deg(x0(3)))
fprintf('Final angle       : %.4f degrees\n',  rad2deg(X(3,end)))
fprintf('Max cart position : %.3f meters\n',   max(abs(X(1,:))))
fprintf('Constraint satisfied: %s\n', ...
         string(max(abs(X(1,:))) <= x_max))

saveas(gcf, 'mpc_pendulum.png')