%% PROJECT 11: Proper Orthogonal Decomposition of a Fluid Flow Wake
% Extracts dominant flow structures using SVD
% Simulates von Karman vortex street behind a cylinder
% Reconstructs flow using only first 2 POD modes

%--- Flow Domain ---
nx    = 100;        % Grid points in x direction
ny    = 50;         % Grid points in y direction
nt    = 200;        % Number of time snapshots
x     = linspace(0, 4*pi, nx);
y     = linspace(-pi, pi, ny);
t     = linspace(0, 20, nt);

[X, Y] = meshgrid(x, y);

fprintf('--- POD Flow Analysis ---\n')
fprintf('Grid size      : %d x %d points\n', nx, ny)
fprintf('Snapshots      : %d\n', nt)
fprintf('Total DOF      : %d\n', nx*ny)

%--- Generate von Karman Vortex Street ---
% Superposition of convecting vortices with shedding frequency
St    = 0.2;        % Strouhal number (shedding frequency)
U_inf = 1.0;        % Freestream velocity

% Velocity field snapshots
U_snapshots = zeros(nx*ny, nt);

fprintf('Generating flow snapshots...\n')

for i = 1:nt
    ti = t(i);

    % Mean flow - simple wake profile
    U_mean = U_inf * (1 - 0.3*exp(-Y.^2/2));

    % First shedding mode - primary vortex street
    mode1  = 0.3*sin(X - U_inf*ti) .* cos(0.5*Y) .* ...
        exp(-0.05*(X-U_inf*ti).^2/pi);

    % Second shedding mode - harmonic
    mode2  = 0.1*sin(2*(X - U_inf*ti)) .* sin(Y) .* ...
        exp(-0.05*(X-U_inf*ti).^2/pi);

    % Random noise - small amplitude
    noise  = 0.02*randn(ny, nx);

    % Total velocity field
    U_field = U_mean + mode1 + mode2 + noise;

    % Store as column vector
    U_snapshots(:,i) = U_field(:);
end

fprintf('Snapshots generated successfully\n')
%--- Compute Mean Flow ---
U_mean_vec = mean(U_snapshots, 2);      % Mean over all snapshots
U_fluct    = U_snapshots - U_mean_vec;  % Fluctuation matrix

%--- Perform SVD ---
fprintf('\nPerforming SVD...\n')
[POD_modes, S, V] = svd(U_fluct, 'econ');

%--- Singular Values and Energy ---
singular_vals  = diag(S);
energy_total   = sum(singular_vals.^2);
energy_frac    = singular_vals.^2 / energy_total * 100;
energy_cumul   = cumsum(energy_frac);

%--- Print Energy Content ---
fprintf('\n--- POD Energy Content ---\n')
fprintf('Mode | Singular Value | Energy %% | Cumulative %%\n')
fprintf('-----|----------------|----------|-------------\n')
for i = 1:10
    fprintf('%4d | %14.4f | %8.2f | %11.2f\n', ...
        i, singular_vals(i), energy_frac(i), energy_cumul(i))
end

%--- Find modes needed for 95% energy ---
n_modes_95 = find(energy_cumul >= 95, 1);
fprintf('\nModes needed for 95%% energy: %d\n', n_modes_95)

%--- Reconstruct Flow with 2 modes ---
n_reconstruct = 2;
U_reconstructed = U_mean_vec + ...
    POD_modes(:,1:n_reconstruct) * ...
    S(1:n_reconstruct,1:n_reconstruct) * ...
    V(:,1:n_reconstruct)';

%--- Reconstruction Error ---
error_2modes = norm(U_fluct - POD_modes(:,1:n_reconstruct)* ...
    S(1:n_reconstruct,1:n_reconstruct)*V(:,1:n_reconstruct)', 'fro');
error_total  = norm(U_fluct, 'fro');
fprintf('Reconstruction error with 2 modes: %.2f%%\n', ...
    error_2modes/error_total*100)
%--- Plot Results ---
figure;

%--- Snapshot of original flow ---
subplot(3,2,1)
U_snap = reshape(U_snapshots(:,50), ny, nx);
contourf(X, Y, U_snap, 20, 'LineStyle', 'none')
colorbar
title('Original Flow Snapshot (t=50)')
xlabel('x'); ylabel('y')
axis tight

%--- Reconstructed flow with 2 modes ---
subplot(3,2,2)
U_recon_snap = reshape(U_reconstructed(:,50), ny, nx);
contourf(X, Y, U_recon_snap, 20, 'LineStyle', 'none')
colorbar
title('Reconstructed Flow - 2 POD Modes')
xlabel('x'); ylabel('y')
axis tight

%--- POD Mode 1 ---
subplot(3,2,3)
mode1_plot = reshape(POD_modes(:,1), ny, nx);
contourf(X, Y, mode1_plot, 20, 'LineStyle', 'none')
colorbar
title(sprintf('POD Mode 1 - %.1f%% Energy', energy_frac(1)))
xlabel('x'); ylabel('y')
axis tight

%--- POD Mode 2 ---
subplot(3,2,4)
mode2_plot = reshape(POD_modes(:,2), ny, nx);
contourf(X, Y, mode2_plot, 20, 'LineStyle', 'none')
colorbar
title(sprintf('POD Mode 2 - %.1f%% Energy', energy_frac(2)))
xlabel('x'); ylabel('y')
axis tight

%--- Energy spectrum ---
subplot(3,2,5)
bar(1:20, energy_frac(1:20), 'FaceColor', 'b')
xlabel('POD Mode Number')
ylabel('Energy (%)')
title('Energy Content per POD Mode')
grid on

%--- Cumulative energy ---
subplot(3,2,6)
plot(1:50, energy_cumul(1:50), 'r-o', 'LineWidth', 2, 'MarkerSize', 4)
yline(95, 'k--', '95% threshold', 'LineWidth', 1.5)
xlabel('Number of Modes')
ylabel('Cumulative Energy (%)')
title('Cumulative Energy Capture')
grid on

%--- Save ---
saveas(gcf, 'pod_flow.png')