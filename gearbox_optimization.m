%% PROJECT 12: Genetic Algorithm Multi-Objective Gearbox Optimization
% Optimizes a 3-stage spur gear train
% Objectives: minimize total mass AND maximize service life
% Variables: number of teeth for each of 6 gears (integers)
% Uses gamultiobj for Pareto front discovery

%--- Gearbox Requirements ---
gear_ratio_total = 20;      % Total gear reduction needed
P_input  = 5000;            % Input power in Watts
n_input  = 1500;            % Input speed in RPM
sigma_b  = 300e6;           % Allowable bending stress in Pa
sigma_c  = 800e6;           % Allowable contact stress in Pa

%--- Gear Material Properties ---
rho      = 7800;            % Steel density in kg/m^3
E_gear   = 200e9;           % Young's modulus in Pa
b_face   = 0.04;            % Face width in meters
m_module = 0.003;           % Module in meters (tooth size)

%--- Design Variable Bounds ---
% Each stage has a driver (small) and driven (large) gear
% Variables: [N1, N2, N3, N4, N5, N6]
% N1,N3,N5 = driver gears (pinions) - fewer teeth
% N2,N4,N6 = driven gears           - more teeth
N_min = 18;         % Minimum teeth (avoid undercutting)
N_max = 120;        % Maximum teeth (size limit)

lb = N_min * ones(1,6);     % Lower bounds
ub = N_max * ones(1,6);     % Upper bounds

fprintf('--- Gearbox Optimization Setup ---\n')
fprintf('Total gear ratio  : %d\n',   gear_ratio_total)
fprintf('Input power       : %.0f W\n',  P_input)
fprintf('Input speed       : %.0f RPM\n', n_input)
fprintf('Tooth range       : [%d, %d]\n', N_min, N_max)
fprintf('Design variables  : 6 gear tooth counts\n')

%--- Fitness Function ---
% Input:  x = [N1, N2, N3, N4, N5, N6] gear tooth counts
% Output: [mass, -life] (GA minimizes both, so negate life)
function f = gearbox_fitness(x, m_module, rho, b_face, ...
    P_input, n_input, sigma_b)

% Round to integers
N = round(x);

% Stage gear ratios
r1 = N(2)/N(1);     % Stage 1 ratio
r2 = N(4)/N(3);     % Stage 2 ratio
r3 = N(6)/N(5);     % Stage 3 ratio

%--- Objective 1: Total Mass ---
% Mass = sum of all gear masses
% Gear mass = rho * pi * (d/2)^2 * b_face
mass = 0;
for i = 1:6
    d_i  = N(i) * m_module;             % Pitch diameter
    mass = mass + rho*pi*(d_i/2)^2*b_face;
end

%--- Objective 2: Service Life ---
% Based on bending stress in the weakest stage
% Torque at each stage
T1 = P_input / (n_input * 2*pi/60);    % Input torque
T2 = T1 * r1;                           % Stage 2 torque
T3 = T2 * r2;                           % Stage 3 torque

% Bending stress at each pinion (Lewis equation simplified)
% sigma = 2T / (b * m * d * Y)  where Y = 0.3 (form factor)
Y  = 0.3;
d1 = N(1)*m_module;
d3 = N(3)*m_module;
d5 = N(5)*m_module;

sig1 = 2*T1 / (b_face * m_module * d1 * Y);
sig2 = 2*T2 / (b_face * m_module * d3 * Y);
sig3 = 2*T3 / (b_face * m_module * d5 * Y);

% Life inversely proportional to max stress ratio
max_stress_ratio = max([sig1, sig2, sig3]) / sigma_b;
life = 1 / max_stress_ratio;   % Higher = longer life

% Return objectives (minimize mass, minimize -life)
f = [mass, -life];

end

%--- Nonlinear Constraints ---
function [c, ceq] = gearbox_constraints(x, gear_ratio_total, ...
    m_module, N_min)
N    = round(x);
r1   = N(2)/N(1);
r2   = N(4)/N(3);
r3   = N(6)/N(5);

% Total ratio must be within 10% of target
ratio_achieved = r1 * r2 * r3;
c(1) =  ratio_achieved - gear_ratio_total*1.1;  % Not too high
c(2) = -ratio_achieved + gear_ratio_total*0.9;  % Not too low

% All ratios must be >= 1 (reduction not increase)
c(3) = 1 - r1;
c(4) = 1 - r2;
c(5) = 1 - r3;

ceq = [];
end

fprintf('Fitness function and constraints defined\n')
%--- GA Options ---
options = optimoptions('gamultiobj', ...
    'PopulationSize',       200, ...
    'MaxGenerations',       100, ...
    'ParetoFraction',       0.5, ...
    'CrossoverFraction',    0.8, ...
    'Display',              'iter', ...
    'PlotFcn',              [], ...
    'UseParallel',          false);

%--- Run Genetic Algorithm ---
fprintf('\nRunning gamultiobj...\n')

fitness_fn  = @(x) gearbox_fitness(x, m_module, rho, b_face, ...
    P_input, n_input, sigma_b);
con_fn      = @(x) gearbox_constraints(x, gear_ratio_total, ...
    m_module, N_min);

IntCon = 1:6;       % All 6 variables must be integers

[x_pareto, f_pareto, flag] = gamultiobj(fitness_fn, 6, ...
    [], [], [], [], lb, ub, con_fn, IntCon, options);

fprintf('\nGA complete. Pareto solutions found: %d\n', size(x_pareto,1))

%--- Extract Objectives ---
mass_pareto = f_pareto(:,1);
life_pareto = -f_pareto(:,2);   % Negate back to positive life

%--- Plot Pareto Front ---
figure;

subplot(1,2,1)
scatter(mass_pareto, life_pareto, 60, 'b', 'filled')
xlabel('Total Mass (kg)')
ylabel('Service Life Index')
title('Pareto Front - Mass vs Service Life')
grid on

% Annotate 3 extreme designs
[~, idx_light]  = min(mass_pareto);
[~, idx_long]   = max(life_pareto);
[~, idx_mid]    = min(abs(mass_pareto - mean(mass_pareto)));

hold on
scatter(mass_pareto(idx_light), life_pareto(idx_light), ...
    150, 'r', 'filled', 'DisplayName', 'Lightest')
scatter(mass_pareto(idx_long),  life_pareto(idx_long), ...
    150, 'g', 'filled', 'DisplayName', 'Longest Life')
scatter(mass_pareto(idx_mid),   life_pareto(idx_mid), ...
    150, 'm', 'filled', 'DisplayName', 'Balanced')
legend('Pareto Solutions', 'Lightest', 'Longest Life', 'Balanced')
hold off

%--- Print 3 Annotated Designs ---
fprintf('\n--- Three Key Pareto Designs ---\n')
fprintf('\nLightest Design:\n')
fprintf('  Teeth : %s\n', num2str(round(x_pareto(idx_light,:))))
fprintf('  Mass  : %.3f kg\n', mass_pareto(idx_light))
fprintf('  Life  : %.3f\n',    life_pareto(idx_light))

fprintf('\nLongest Life Design:\n')
fprintf('  Teeth : %s\n', num2str(round(x_pareto(idx_long,:))))
fprintf('  Mass  : %.3f kg\n', mass_pareto(idx_long))
fprintf('  Life  : %.3f\n',    life_pareto(idx_long))

fprintf('\nBalanced Design:\n')
fprintf('  Teeth : %s\n', num2str(round(x_pareto(idx_mid,:))))
fprintf('  Mass  : %.3f kg\n', mass_pareto(idx_mid))
fprintf('  Life  : %.3f\n',    life_pareto(idx_mid))

%--- Subplot 2: Tooth counts for 3 designs ---
subplot(1,2,2)
designs = [round(x_pareto(idx_light,:));
    round(x_pareto(idx_mid,:));
    round(x_pareto(idx_long,:))];
bar(designs')
xlabel('Gear Number')
ylabel('Number of Teeth')
title('Tooth Counts for 3 Key Designs')
legend('Lightest', 'Balanced', 'Longest Life')
set(gca, 'XTickLabel', {'N1','N2','N3','N4','N5','N6'})
grid on

saveas(gcf, 'gearbox_optimization.png')
