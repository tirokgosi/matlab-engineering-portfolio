%% PROJECT 20: Model Identification of a DC Motor using Subspace Methods
% Simulates a noisy DC motor driven by PRBS input
% Identifies state-space model directly from input-output data
% Compares identified model to true motor via Bode plot

%--- True DC Motor Parameters ---
K_motor = 0.5;      % Motor gain constant
tau     = 0.1;      % Motor time constant in seconds
% Transfer function: G(s) = K/(tau*s + 1)
G_true  = tf(K_motor, [tau, 1]);

fprintf('--- DC Motor System Identification ---\n')
fprintf('True motor gain     : %.2f\n', K_motor)
fprintf('True time constant  : %.3f seconds\n', tau)

%--- Generate PRBS Input Signal ---
rng(42)
dt      = 0.01;         % Sample time in seconds
t_end   = 20;           % Experiment duration
t       = 0:dt:t_end;
N       = length(t);

% Pseudo-Random Binary Signal
% Switches randomly between -1 and +1
prbs    = sign(randn(1,N));
prbs    = prbs * 5;     % Scale to ±5 Volts

%--- Simulate True Motor Response ---
sys_d   = c2d(G_true, dt, 'zoh');      % Discretize true plant
[y_true, ~] = lsim(sys_d, prbs, t);    % Simulate response

%--- Add Measurement Noise ---
noise_std = 0.1;
y_noisy   = y_true + noise_std*randn(N,1);

fprintf('PRBS signal amplitude : ±5 V\n')
fprintf('Measurement noise std : %.2f\n', noise_std)
fprintf('Experiment duration   : %.0f seconds\n', t_end)
fprintf('Sample time           : %.3f seconds\n', dt)

%--- System Identification using n4sid ---
fprintf('\nRunning n4sid subspace identification...\n')

% Create iddata object
data = iddata(y_noisy, prbs', dt);

% Identify model - let MATLAB choose order automatically
sys_id = n4sid(data, 'best', 'Ts', dt);

fprintf('Identified model order: %d\n', order(sys_id))

% Convert to transfer function for comparison
G_id = tf(sys_id);

%--- Extract Identified Time Constant ---
p_id  = pole(G_id);
tau_id = -1/real(p_id(1))*dt;

fprintf('\n--- Identification Results ---\n')
fprintf('True time constant    : %.4f seconds\n', tau)
fprintf('Identified time const : %.4f seconds\n', abs(1/real(p_id(1))*dt))

%--- Simulate Identified Model ---
[y_id, ~] = lsim(sys_id, prbs', t);

%--- Validation: Compare step responses ---
t_step          = 0:dt:2;
[y_step_true,~] = step(sys_d,  t_step);
[y_step_id,  ~] = step(sys_id, t_step);

%--- Bode Plot Comparison ---
omega = logspace(-1, 3, 500);
[mag_true, phase_true] = bode(G_true, omega);
[mag_id,   phase_id  ] = bode(G_id,   omega);

mag_true  = squeeze(mag_true);
mag_id    = squeeze(mag_id);
phase_true = squeeze(phase_true);
phase_id   = squeeze(phase_id);

%--- Fit percentage ---
fit_pct = 100*(1 - norm(y_true - y_id)/norm(y_true - mean(y_true)));
fprintf('Model fit percentage  : %.1f%%\n', fit_pct)

%--- Plot ---
figure;

%--- Input signal ---
subplot(3,2,1)
plot(t, prbs, 'b-', 'LineWidth', 1)
xlabel('Time (s)')
ylabel('Voltage (V)')
title('PRBS Input Signal')
grid on

%--- Output comparison ---
subplot(3,2,2)
plot(t, y_noisy, 'r.', 'MarkerSize', 2)
hold on
plot(t, y_id,    'b-', 'LineWidth', 1.5)
plot(t, y_true,  'k-', 'LineWidth', 1.5)
legend({'Noisy output','Identified model','True output'}, ...
        'Location', 'best')
title('Output Comparison')
xlabel('Time (s)')
ylabel('Speed (rad/s)')
grid on
hold off

%--- Step response comparison ---
subplot(3,2,3)
plot(t_step, y_step_true, 'k-', 'LineWidth', 2)
hold on
plot(t_step, y_step_id,   'b--', 'LineWidth', 2)
legend({'True motor','Identified model'}, 'Location', 'best')
title('Step Response Comparison')
xlabel('Time (s)')
ylabel('Speed (rad/s)')
grid on
hold off

%--- Bode magnitude ---
subplot(3,2,4)
semilogx(omega, 20*log10(mag_true), 'k-', 'LineWidth', 2)
hold on
semilogx(omega, 20*log10(mag_id),   'b--', 'LineWidth', 2)
legend({'True motor','Identified model'}, 'Location', 'best')
title('Bode Plot - Magnitude')
xlabel('Frequency (rad/s)')
ylabel('Magnitude (dB)')
grid on
hold off

%--- Bode phase ---
subplot(3,2,5)
semilogx(omega, phase_true, 'k-', 'LineWidth', 2)
hold on
semilogx(omega, phase_id,   'b--', 'LineWidth', 2)
legend({'True motor','Identified model'}, 'Location', 'best')
title('Bode Plot - Phase')
xlabel('Frequency (rad/s)')
ylabel('Phase (degrees)')
grid on
hold off

%--- Residuals ---
subplot(3,2,6)
residuals = y_true - y_id;
plot(t, residuals, 'm-', 'LineWidth', 1)
yline(0, 'k--', 'LineWidth', 1)
xlabel('Time (s)')
ylabel('Error')
title(sprintf('Residuals - Fit = %.1f%%', fit_pct))
grid on

sgtitle('DC Motor System Identification using n4sid')

saveas(gcf, 'motor_identification.png')