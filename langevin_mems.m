%% PROJECT 15: Langevin Dynamics for a Bistable MEMS Switch
% Models a micro-switch as a particle in a double-well potential
% Shows noise-induced switching (stochastic resonance)
% Euler-Maruyama integration of the Langevin equation

%--- Potential Parameters ---
a = 1.0;        % Quartic coefficient (well shape)
b = 2.0;        % Quadratic coefficient (well depth)

% Double well potential: U(x) = a*x^4 - b*x^2
% Stable equilibria at x = +/-sqrt(b/(2a)) = +/-1
% Barrier height = b^2/(4a) = 1.0

U  = @(x)  a*x.^4 - b*x.^2;           % Potential energy
dU = @(x) 4*a*x.^3 - 2*b*x;           % Force = -dU/dx

%--- Simulation Parameters ---
dt      = 0.001;        % Time step
t_end   = 50;           % Total simulation time
t       = 0:dt:t_end;
N       = length(t);

%--- Signal Parameters ---
% Weak periodic signal - not strong enough alone to cause switching
A_sig   = 0.5;          % Signal amplitude (below switching threshold)
omega   = 0.5;          % Signal frequency

%--- Noise Parameters ---
sigma   = 0.8;          % Noise amplitude
% Too low: no switching. Too high: random jumping. Just right: resonance

%--- Initial Condition ---
x0      = -1.0;         % Start in left well

%--- Print setup ---
fprintf('--- Langevin MEMS Switch Simulation ---\n')
fprintf('Potential     : U(x) = %.1fx^4 - %.1fx^2\n', a, b)
fprintf('Stable states : x = +/-%.2f\n', sqrt(b/(2*a)))
fprintf('Barrier height: %.2f\n', b^2/(4*a))
fprintf('Signal amp    : %.1f (below threshold)\n', A_sig)
fprintf('Noise amp     : %.1f\n', sigma)
fprintf('Time steps    : %d\n', N)
%--- Euler-Maruyama Integration ---
% Langevin equation: dx = -dU/dx*dt + A_sig*sin(omega*t)*dt + sigma*dW
% dW = sqrt(dt)*randn  (Wiener process increment)

rng(42)                 % Fix random seed
x      = zeros(1,N);   % Position trajectory
x(1)   = x0;           % Initial condition

fprintf('\nRunning Euler-Maruyama integration...\n')

for i = 1:N-1

    ti = t(i);

    % Deterministic drift: restoring force + signal
    drift = -dU(x(i)) + A_sig*sin(omega*ti);

    % Stochastic diffusion: thermal noise
    dW    = sqrt(dt)*randn;
    noise = sigma * dW;

    % Euler-Maruyama update
    x(i+1) = x(i) + drift*dt + noise;

end

%--- Count switching events ---
switches = 0;
in_right = x(1) > 0;
for i = 2:N
    now_right = x(i) > 0;
    if now_right ~= in_right
        switches  = switches + 1;
        in_right  = now_right;
    end
end

fprintf('Integration complete\n')
fprintf('Switching events detected: %d\n', switches)
fprintf('Time in left  well: %.1f%%\n', 100*sum(x<0)/N)
fprintf('Time in right well: %.1f%%\n', 100*sum(x>0)/N)
%--- Plot Results ---
figure;

%--- Potential Well Plot ---
subplot(3,1,1)
x_pot = linspace(-1.8, 1.8, 500);
plot(x_pot, U(x_pot), 'k-', 'LineWidth', 2.5)
hold on

% Mark stable equilibria
plot(-1, U(-1), 'bs', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'b', 'DisplayName', 'Left well')
plot( 1, U( 1), 'rs', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'r', 'DisplayName', 'Right well')
plot( 0, U( 0), 'k^', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'k', 'DisplayName', 'Unstable top')

% Mark barrier height
yline(0, 'k--', 'Barrier level', 'LineWidth', 1)
legend('Potential U(x)', 'Left well', 'Right well', ...
    'Unstable equilibrium')
title('Double Well Potential U(x) = x^4 - 2x^2')
xlabel('Position x')
ylabel('Potential Energy U(x)')
xlim([-1.8, 1.8])
ylim([-1.5, 1.5])
grid on
hold off

%--- Signal Plot ---
subplot(3,1,2)
signal = A_sig*sin(omega*t);
plot(t, signal, 'g-', 'LineWidth', 1.5)
yline( 1, 'r--', 'Barrier', 'LineWidth', 1.5)
yline(-1, 'r--', 'LineWidth', 1.5)
title(sprintf('Weak Periodic Signal (Amplitude=%.1f, Barrier=1.0)', A_sig))
xlabel('Time')
ylabel('Signal')
ylim([-1.5, 1.5])
grid on

%--- Trajectory Plot ---
subplot(3,1,3)
plot(t, x, 'b-', 'LineWidth', 0.8)
hold on
yline( 1, 'r--', 'Right well', 'LineWidth', 1.5)
yline(-1, 'b--', 'Left well',  'LineWidth', 1.5)
yline( 0, 'k--', 'Barrier',    'LineWidth', 1)
title(sprintf('Langevin Trajectory - %d Switching Events', switches))
xlabel('Time')
ylabel('Position x(t)')
ylim([-2.5, 2.5])
grid on
hold off

%--- Print insight ---
fprintf('\n--- Insight ---\n')
fprintf('Signal alone (amp=%.1f) cannot overcome barrier (height=1.0)\n', ...
    A_sig)
fprintf('Noise (sigma=%.1f) enables %d switching events\n', sigma, switches)
fprintf('This is stochastic resonance - noise aids signal detection\n')

%--- Save ---
saveas(gcf, 'langevin_mems.png')

