%% PROJECT 19: Fatigue Life Prediction with Rainflow Counting
% Generates random stress signal and predicts fatigue life
% Uses rainflow counting and Palmgren-Miner damage rule
% S-N curve for steel material

%--- Material Properties (Steel) ---
S_f  = 600e6;       % Fatigue strength coefficient in Pa
b    = -0.085;      % Fatigue strength exponent
E    = 200e9;       % Young's modulus in Pa
UTS  = 800e6;       % Ultimate tensile strength in Pa

%--- S-N Curve Function ---
% S = S_f * (2*N)^b  -> N = 0.5*(S/S_f)^(1/b)
cycles_to_failure = @(S) 0.5*(S/S_f).^(1/b);

%--- Generate Random Stress Signal ---
rng(42)
dt      = 0.01;         % Time step in seconds
t_end   = 100;          % Signal length in seconds
t       = 0:dt:t_end;
N_pts   = length(t);

% Realistic aircraft gust response signal
% Superposition of multiple frequency components
sigma = 50e6*sin(2*pi*0.5*t)  + ...   % Low frequency gust
        30e6*sin(2*pi*2.0*t)  + ...   % Medium frequency
        20e6*sin(2*pi*5.0*t)  + ...   % Higher frequency
        40e6*randn(1,N_pts);           % Random turbulence

% Scale to realistic stress range
sigma = sigma / max(abs(sigma)) * 250e6;   % Peak stress 250 MPa

fprintf('--- Fatigue Life Prediction ---\n')
fprintf('Material         : Steel\n')
fprintf('UTS              : %.0f MPa\n', UTS/1e6)
fprintf('Fatigue strength : %.0f MPa\n', S_f/1e6)
fprintf('Signal length    : %.0f seconds\n', t_end)
fprintf('Data points      : %d\n', N_pts)
fprintf('Peak stress      : %.1f MPa\n', max(abs(sigma))/1e6)

%--- Rainflow Counting ---
% Extract peaks and valleys first
fprintf('\nPerforming rainflow counting...\n')

% Find turning points (peaks and valleys)
d    = diff(sigma);
idx  = find(diff(sign(d)) ~= 0) + 1;
idx  = [1, idx, length(sigma)];
sig_tp = sigma(idx);       % Turning points only

% Simple rainflow counting implementation
% Based on ASTM E1049 standard
n_tp   = length(sig_tp);
stack  = [];
cycles = [];

for i = 1:n_tp
    stack = [stack, sig_tp(i)];

    % Keep extracting cycles while possible
    while length(stack) >= 3
        n_stack = length(stack);
        S1 = abs(stack(n_stack-1) - stack(n_stack-2));
        S2 = abs(stack(n_stack)   - stack(n_stack-1));

        if S2 >= S1
            % Extract cycle with range S1
            range_cycle = S1;
            mean_cycle  = (stack(n_stack-1) + stack(n_stack-2))/2;
            cycles      = [cycles; range_cycle, mean_cycle];
            % Remove middle two points
            stack(n_stack-2:n_stack-1) = [];
        else
            break
        end
    end
end

% Remaining half cycles
for i = 1:length(stack)-1
    range_cycle = abs(stack(i+1) - stack(i));
    mean_cycle  = (stack(i+1) + stack(i))/2;
    cycles      = [cycles; range_cycle/2, mean_cycle];
end

fprintf('Turning points found : %d\n', n_tp)
fprintf('Cycles extracted     : %d\n', size(cycles,1))

%--- Palmgren-Miner Damage Rule ---
% D = sum(n_i / N_fi)
% n_i = number of cycles at stress amplitude S_i
% N_fi = cycles to failure at S_i from S-N curve

stress_amplitudes = cycles(:,1)/2;     % Amplitude = range/2
mean_stresses     = cycles(:,2);

% Apply Goodman mean stress correction
% S_corrected = S_a / (1 - S_m/UTS)
S_corrected = stress_amplitudes ./ (1 - mean_stresses/UTS);
S_corrected = max(S_corrected, 0);     % No negative stress

% Cycles to failure for each cycle
N_fail = cycles_to_failure(max(S_corrected, 1e6));  % Min 1 MPa

% Damage per cycle
damage_per_cycle = 1 ./ N_fail;

% Total damage per mission profile
D_total = sum(damage_per_cycle);

% Predicted life
life_profiles = 1 / D_total;

fprintf('\n--- Damage Analysis ---\n')
fprintf('Total cycles per profile : %d\n', size(cycles,1))
fprintf('Cumulative damage D      : %.6f\n', D_total)
fprintf('Predicted life           : %.0f mission profiles\n', life_profiles)

if D_total < 1
    fprintf('Component is SAFE for one mission profile\n')
else
    fprintf('WARNING: Component FAILS within one mission profile\n')
end

%--- Plot ---
figure;

%--- Stress Signal ---
subplot(2,2,1)
plot(t, sigma/1e6, 'b-', 'LineWidth', 0.8)
xlabel('Time (seconds)')
ylabel('Stress (MPa)')
title('Random Stress Signal - Aircraft Gust Response')
grid on

%--- Rainflow Matrix ---
subplot(2,2,2)
scatter(mean_stresses/1e6, stress_amplitudes/1e6, ...
        20, damage_per_cycle, 'filled')
colorbar
colormap('jet')
xlabel('Mean Stress (MPa)')
ylabel('Stress Amplitude (MPa)')
title('Rainflow Matrix - Colored by Damage')
grid on

%--- S-N Curve ---
subplot(2,2,3)
N_plot = logspace(3, 8, 500);
S_plot = S_f * (2*N_plot).^b;
loglog(N_plot, S_plot/1e6, 'k-', 'LineWidth', 2)
hold on
scatter(N_fail, stress_amplitudes/1e6, 20, 'r', 'filled')
xlabel('Cycles to Failure N')
ylabel('Stress Amplitude (MPa)')
title('S-N Curve with Cycle Distribution')
legend({'S-N curve','Counted cycles'}, 'Location', 'best')
grid on
hold off

%--- Damage Histogram ---
subplot(2,2,4)
[counts, edges] = histcounts(stress_amplitudes/1e6, 20);
bar(edges(1:end-1), counts, 'FaceColor', 'b', 'EdgeColor', 'none')
xlabel('Stress Amplitude (MPa)')
ylabel('Cycle Count')
title('Stress Amplitude Distribution')
grid on

sgtitle(sprintf('Fatigue Life: %.0f Mission Profiles | D = %.4f', ...
                 life_profiles, D_total))

saveas(gcf, 'fatigue_rainflow.png')