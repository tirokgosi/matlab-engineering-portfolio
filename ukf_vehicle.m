%% PROJECT 9: Unscented Kalman Filter for Vehicle State Estimation
% Simulates a car driving a figure-8 pattern
% State: [x, y, theta, v] - position, heading, speed
% Measurements: noisy GPS (x,y) and noisy speed sensor
% UKF handles nonlinear heading dynamics without Jacobian

%--- Simulation Parameters ---
dt = 0.1;           % Time step in seconds
t  = 0:dt:20;       % 20 seconds total
N  = length(t);     % Number of time steps

%--- True Path: Figure-8 ---
x_true     = 50*sin(2*pi*t/20);
y_true     = 50*sin(2*pi*t/10);
dx         = gradient(x_true, dt);
dy         = gradient(y_true, dt);
theta_true = atan2(dy, dx);
v_true     = sqrt(dx.^2 + dy.^2);
omega      = gradient(theta_true, dt);    % Heading rate

fprintf('--- UKF Vehicle State Estimation ---\n')
fprintf('Simulation time : %.0f seconds\n', t(end))
fprintf('Time steps      : %d\n', N)

%--- Measurement Noise ---
rng(42)
gps_std   = 3.0;
speed_std = 0.5;

z_x = x_true + gps_std   * randn(1,N);
z_y = y_true + gps_std   * randn(1,N);
z_v = v_true + speed_std * randn(1,N);
Z   = [z_x; z_y; z_v];

%--- Noise Covariances ---
Q = diag([0.01, 0.01, 0.001, 0.01].^2);
R = diag([gps_std, gps_std, speed_std].^2);

fprintf('\n--- Noise Parameters ---\n')
fprintf('GPS noise std   : %.1f meters\n', gps_std)
fprintf('Speed noise std : %.1f m/s\n', speed_std)

%--- UKF Parameters ---
n      = 4;
m      = 3;
alpha  = 1e-3;
kappa  = 0;
beta   = 2;
lambda = alpha^2*(n+kappa) - n;

%--- Sigma Point Weights ---
Wm    = zeros(1, 2*n+1);
Wc    = zeros(1, 2*n+1);
Wm(1) = lambda/(n+lambda);
Wc(1) = lambda/(n+lambda) + (1-alpha^2+beta);
for i = 2:2*n+1
    Wm(i) = 1/(2*(n+lambda));
    Wc(i) = 1/(2*(n+lambda));
end

%--- Measurement Model ---
h = @(x) [x(1); x(2); x(4)];

%--- Initialize ---
x_est      = zeros(n,N);
x_est(:,1) = [z_x(1); z_y(1); theta_true(1); v_true(1)];
P          = diag([gps_std^2, gps_std^2, 0.1, speed_std^2]);

fprintf('\nRunning UKF...\n')

%--- UKF Loop ---
for k = 2:N

    %--- Motion model at this time step ---
    f = @(x) [x(1) + x(4)*cos(x(3))*dt;
               x(2) + x(4)*sin(x(3))*dt;
               x(3) + omega(k)*dt;
               x(4)];

    %--- Generate Sigma Points ---
    S     = chol((n+lambda)*P + eye(n)*1e-6, 'lower');
    Xsig  = [x_est(:,k-1), ...
             x_est(:,k-1)*ones(1,n) + S, ...
             x_est(:,k-1)*ones(1,n) - S];

    %--- Propagate Sigma Points ---
    Xsig_pred = zeros(n, 2*n+1);
    for i = 1:2*n+1
        Xsig_pred(:,i) = f(Xsig(:,i));
    end

    %--- Predicted Mean and Covariance ---
    x_pred = Xsig_pred * Wm';
    P_pred = Q;
    for i = 1:2*n+1
        diff   = Xsig_pred(:,i) - x_pred;
        P_pred = P_pred + Wc(i)*(diff*diff');
    end

    %--- Measurement Sigma Points ---
    Zsig = zeros(m, 2*n+1);
    for i = 1:2*n+1
        Zsig(:,i) = h(Xsig_pred(:,i));
    end
    z_pred = Zsig * Wm';

    %--- Innovation and Cross Covariance ---
    Pzz = R;
    Pxz = zeros(n,m);
    for i = 1:2*n+1
        dz  = Zsig(:,i)      - z_pred;
        dxx = Xsig_pred(:,i) - x_pred;
        Pzz = Pzz + Wc(i)*(dz*dz');
        Pxz = Pxz + Wc(i)*(dxx*dz');
    end

    %--- Kalman Gain and Update ---
    K             = Pxz / Pzz;
    innovation    = Z(:,k) - z_pred;
    x_est(:,k)   = x_pred + K*innovation;
    P             = P_pred - K*Pzz*K';
    P             = (P+P')/2;

end

fprintf('UKF complete\n')

%--- Plot ---
figure;

subplot(2,2,[1,2])
hold on
scatter(z_x, z_y, 8, 'r', 'filled', 'DisplayName', 'Noisy GPS')
plot(x_true, y_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Path')
plot(x_est(1,:), x_est(2,:), 'b-', 'LineWidth', 2, 'DisplayName', 'UKF Estimate')
plot(x_true(1), y_true(1), 'gs', 'MarkerSize', 12, ...
     'MarkerFaceColor', 'g', 'DisplayName', 'Start')
legend('Location', 'northeast')
title('UKF Vehicle State Estimation - Figure-8 Path')
xlabel('x position (m)')
ylabel('y position (m)')
grid on
axis equal
hold off

subplot(2,2,3)
plot(t, rad2deg(theta_true), 'k-', 'LineWidth', 2)
hold on
plot(t, rad2deg(x_est(3,:)), 'b-', 'LineWidth', 1.5)
legend('True Heading', 'UKF Estimate')
title('Heading Angle Estimation')
xlabel('Time (seconds)')
ylabel('Heading (degrees)')
grid on
hold off

subplot(2,2,4)
plot(t, v_true, 'k-', 'LineWidth', 2)
hold on
plot(t, z_v,        'r.', 'MarkerSize', 4)
plot(t, x_est(4,:), 'b-', 'LineWidth', 1.5)
legend('True Speed', 'Noisy Sensor', 'UKF Estimate')
title('Speed Estimation')
xlabel('Time (seconds)')
ylabel('Speed (m/s)')
grid on
hold off

%--- Performance ---
pos_error_gps = mean(sqrt((z_x-x_true).^2 + (z_y-y_true).^2));
pos_error_ukf = mean(sqrt((x_est(1,:)-x_true).^2 + ...
                          (x_est(2,:)-y_true).^2));

fprintf('\n--- Performance ---\n')
fprintf('Mean GPS position error : %.2f meters\n', pos_error_gps)
fprintf('Mean UKF position error : %.2f meters\n', pos_error_ukf)
fprintf('UKF improvement         : %.1fx better than GPS\n', ...
         pos_error_gps/pos_error_ukf)

saveas(gcf, 'ukf_vehicle.png')