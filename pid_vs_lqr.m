%% PROJECT 3: PID vs LQR on a Mass-Spring-Damper
% Compares a PID controller and an LQR controller
% System: mass m=1kg, spring k=10 N/m, damping b=0.5 Ns/m

%--- System Parameters ---
m = 1;      % Mass in kg
k = 10;     % Spring stiffness in N/m
b = 0.5;    % Damping coefficient in Ns/m

%--- Transfer Function representation (for PID) ---
% G(s) = 1 / (ms^2 + bs + k)
% This describes how the mass responds to a force input
num = [1];                  % Numerator coefficients
den = [m, b, k];            % Denominator coefficients [m, b, k]
G = tf(num, den);           % Create transfer function object

%--- State Space representation (for LQR) ---
% State vector: x = [position, velocity]
% dx/dt = Ax + Bu
% y = Cx + Du
A = [0,    1;               % Position changes with velocity
    -k/m, -b/m];            % Velocity changes with force and damping
B = [0; 1/m];               % Force input affects velocity
C = [1, 0];                 % We measure position only
D = 0;                      % No direct feedthrough
sys = ss(A, B, C, D);       % Create state space object
%--- PID Controller Design ---
% pidtune automatically finds good PID gains for our system
C_pid = pidtune(G, 'PID');        % Auto tune PID for transfer function G

% Print the PID gains
fprintf('\n--- PID Gains ---\n')
fprintf('Kp = %.4f\n', C_pid.Kp)  % Proportional gain
fprintf('Ki = %.4f\n', C_pid.Ki)  % Integral gain
fprintf('Kd = %.4f\n', C_pid.Kd) % Derivative gain

% Create closed loop system with PID controller
sys_pid = feedback(C_pid * G, 1);

% Simulate step response
t = linspace(0, 10, 1000);        % Time vector 0 to 10 seconds
[y_pid, t_pid] = step(sys_pid, t);% Simulate step response
%--- LQR Controller Design ---
% Q penalizes position error - how much we care about accuracy
% R penalizes control effort - how much we care about energy used
Q = [100, 0;     % Penalize position error heavily
    0,  1];    % Penalize velocity error lightly
R = 1;           % Penalize control effort moderately

% Calculate optimal LQR gain matrix
K_lqr = lqr(A, B, Q, R);

fprintf('\n--- LQR Gains ---\n')
fprintf('K1 (position gain) = %.4f\n', K_lqr(1))
fprintf('K2 (velocity gain) = %.4f\n', K_lqr(2))

%--- Simulate LQR Step Response ---
% For LQR we simulate manually using the closed loop A matrix
A_cl = A - B*K_lqr;             % Closed loop system matrix

% We need to add a reference input to track position = 1m
% Compute feedforward gain
Nbar = -1/(C*((A_cl)\B));       % Feedforward scaling gain

% Simulate using lsim
u_ref = ones(1, length(t));             % Step reference: go to position 1m
U_lqr = Nbar * u_ref;                  % Scale input
[y_lqr, t_lqr] = lsim(ss(A_cl, B, C, D), U_lqr, t);
%--- Plot PID vs LQR Step Response ---
figure;
hold on;

% Plot both responses
plot(t_pid, y_pid, 'b-', 'LineWidth', 2)
plot(t_lqr, y_lqr, 'r-', 'LineWidth', 2)

% Reference line at position = 1m
yline(1, 'k--', 'Target Position', 'LineWidth', 1.5)

% Calculate settling time and overshoot for PID
pid_overshoot = (max(y_pid) - 1) * 100;
lqr_overshoot = (max(y_lqr) - 1) * 100;

% Annotations
fprintf('\n--- Performance Comparison ---\n')
fprintf('PID Overshoot  = %.2f%%\n', pid_overshoot)
fprintf('LQR Overshoot  = %.2f%%\n', lqr_overshoot)

% Labels
legend('PID Controller', 'LQR Controller')
title('PID vs LQR - Mass Spring Damper Step Response')
xlabel('Time (seconds)')
ylabel('Position (m)')
grid on
hold off

% Save figure
saveas(gcf, 'pid_vs_lqr.png')