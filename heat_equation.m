%% PROJECT 2: 1D Transient Heat Equation Solver
% Models heat diffusion through a rod over time
% Left end: fixed at 100 degrees C
% Right end: insulated (no heat escapes)

%--- Rod Properties ---
L = 1.0;          % Length of rod in meters
alpha = 1e-4;     % Thermal diffusivity (m^2/s) - typical for steel
T_left = 100;     % Temperature at left end in degrees C
T_init = 0;       % Initial temperature of entire rod in degrees C

%--- Discretization ---
nx = 20;                        % Number of segments along the rod
dx = L/nx;                      % Size of each segment in meters
x = linspace(0, L, nx+1);      % x positions of all nodes (21 points)

%--- Stability Condition ---
% For explicit finite difference the Fourier number must be <= 0.5
% Fourier number = alpha * dt / dx^2
% So maximum stable dt = 0.5 * dx^2 / alpha
dt = 0.5 * dx^2 / alpha;       % Maximum stable time step in seconds
t_end = 5000;                   % Total simulation time in seconds
nt = floor(t_end/dt);           % Number of time steps

%--- Initialize Temperature Array ---
T = ones(1, nx+1) * T_init;    % All nodes start at 0 degrees C
T(1) = T_left;                  % Left end fixed at 100 degrees C

%--- Pre-allocate storage for plotting ---
% We dont store every time step - just every 100th one
plot_every = 100;
T_history = zeros(floor(nt/plot_every), nx+1);
t_history = zeros(floor(nt/plot_every), 1);
counter = 1;

%--- Time Loop ---
for n = 1:nt

    T_new = T;     % Copy current temperature to work with

    % Update every interior node using explicit finite difference
    for i = 2:nx
        T_new(i) = T(i) + alpha * dt/dx^2 * (T(i+1) - 2*T(i) + T(i-1));
    end

    % Insulated right end: mirror condition
    % No heat flows out so we treat it as if there's a mirror node
    T_new(nx+1) = T_new(nx);

    % Left end stays fixed always
    T_new(1) = T_left;

    % Update temperature
    T = T_new;

    % Save every 100th step for plotting
    if mod(n, plot_every) == 0
        T_history(counter, :) = T;
        t_history(counter) = n * dt;
        counter = counter + 1;
    end

end

%--- Plot Temperature Evolution ---
figure;

% Waterfall plot - each line is a snapshot in time
waterfall(x, t_history(1:counter-1), T_history(1:counter-1, :))

% Labels
xlabel('Position along rod (m)')
ylabel('Time (seconds)')
zlabel('Temperature (degrees C)')
title('1D Transient Heat Equation - Temperature Evolution')
colormap('hot')
view(45, 30)    % Viewing angle for 3D plot
grid on

% Print stability information
fprintf('\n--- Simulation Parameters ---\n')
fprintf('Time step dt     = %.4f seconds\n', dt)
fprintf('Total time       = %.0f seconds\n', t_end)
fprintf('Number of steps  = %d\n', nt)
fprintf('Fourier number   = %.4f (must be <= 0.5)\n', alpha*dt/dx^2)
saveas(gcf, 'heat_equation.png')