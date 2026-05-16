%% PROJECT 24: Lattice Boltzmann Method - 2D Channel Flow
% D2Q9 lattice for pressure-driven Poiseuille flow
% BGK collision operator
% Bounce-back boundary conditions on walls
% Compares to analytical Poiseuille solution

%--- LBM Parameters ---
nx    = 100;        % Grid points in x
ny    = 40;         % Grid points in y
omega = 1.7;        % Relaxation parameter (viscosity)
nu    = (1/omega - 0.5)/3;  % Kinematic viscosity
rho0  = 1.0;        % Reference density
F     = 0.0001;     % Body force (pressure gradient)
n_steps = 5000;     % Number of time steps

fprintf('--- Lattice Boltzmann Channel Flow ---\n')
fprintf('Grid          : %d x %d\n', nx, ny)
fprintf('Relaxation    : %.2f\n', omega)
fprintf('Viscosity     : %.6f\n', nu)
fprintf('Body force    : %.4f\n', F)
fprintf('Time steps    : %d\n', n_steps)

%--- D2Q9 Lattice ---
% Velocity directions
ex = [0, 1, 0,-1, 0, 1,-1,-1, 1];
ey = [0, 0, 1, 0,-1, 1, 1,-1,-1];
w  = [4/9, 1/9, 1/9, 1/9, 1/9, ...
      1/36,1/36,1/36,1/36];

% Opposite directions for bounce-back
opp = [1, 4, 5, 2, 3, 8, 9, 6, 7];

%--- Initialize Distribution Functions ---
f    = zeros(9, nx, ny);
feq  = zeros(9, nx, ny);

for k = 1:9
    f(k,:,:) = w(k)*rho0;
end

%--- Macroscopic Variables ---
rho  = zeros(nx, ny);
ux   = zeros(nx, ny);
uy   = zeros(nx, ny);

fprintf('\nRunning LBM simulation...\n')

%--- Main LBM Loop ---
for step = 1:n_steps

    %--- Compute Macroscopic Variables ---
    rho = squeeze(sum(f, 1));
    ux  = zeros(nx, ny);
    uy  = zeros(nx, ny);

    for k = 1:9
        ux = ux + ex(k)*squeeze(f(k,:,:));
        uy = uy + ey(k)*squeeze(f(k,:,:));
    end
    ux = ux./rho;
    uy = uy./rho;

    %--- Add Body Force ---
    ux = ux + F/2;

    %--- Compute Equilibrium ---
    for k = 1:9
        eu  = ex(k)*ux + ey(k)*uy;
        feq(k,:,:) = w(k)*rho.*(1 + 3*eu + 4.5*eu.^2 - 1.5*(ux.^2+uy.^2));
    end

    %--- Collision (BGK) ---
    f = f - omega*(f - feq);

    %--- Add Force to Distribution ---
    for k = 1:9
        eu = ex(k)*F;
        f(k,:,:) = f(k,:,:) + w(k)*3*eu;
    end

    %--- Streaming ---
    for k = 1:9
        f(k,:,:) = circshift(squeeze(f(k,:,:)), [ex(k), ey(k)]);
    end

    %--- Bounce-Back on Top and Bottom Walls ---
    % Bottom wall (j=1)
    for k = 1:9
        if ey(k) < 0
            f(k,:,1) = f(opp(k),:,1);
        end
    end
    % Top wall (j=ny)
    for k = 1:9
        if ey(k) > 0
            f(k,:,ny) = f(opp(k),:,ny);
        end
    end

    %--- Periodic BC in x (already handled by circshift) ---

    if mod(step, 1000) == 0
        fprintf('Step %d | Max ux = %.6f\n', step, max(max(ux)))
    end

end

fprintf('LBM simulation complete\n')

%--- Extract Velocity Profile at Center ---
x_mid   = round(nx/2);
ux_prof = squeeze(ux(x_mid,:));
y_pts   = 1:ny;

%--- Analytical Poiseuille Solution ---
H       = ny - 1;
y_norm  = (y_pts - 1)/H;
U_max   = F*H^2/(8*nu);
ux_anal = 4*U_max*y_norm.*(1-y_norm);

%--- Print Comparison ---
fprintf('\n--- Validation ---\n')
fprintf('Max numerical velocity : %.6f\n', max(ux_prof))
fprintf('Max analytical velocity: %.6f\n', max(ux_anal))
fprintf('Error                  : %.4f%%\n', ...
         abs(max(ux_prof)-max(ux_anal))/max(ux_anal)*100)

%--- Plot ---
figure;

subplot(1,3,1)
contourf(squeeze(ux)', 20, 'LineStyle', 'none')
colorbar
colormap('jet')
title('x-Velocity Field ux')
xlabel('x'); ylabel('y')
axis tight

subplot(1,3,2)
plot(ux_prof, y_pts, 'b-o', 'LineWidth', 2, 'MarkerSize', 4)
hold on
plot(ux_anal, y_pts, 'r--', 'LineWidth', 2)
legend({'LBM numerical','Analytical Poiseuille'}, 'Location', 'best')
title('Velocity Profile at Channel Center')
xlabel('u_x velocity')
ylabel('y position')
grid on
hold off

subplot(1,3,3)
error_pct = abs(ux_prof - ux_anal)./max(ux_anal)*100;
plot(error_pct, y_pts, 'k-', 'LineWidth', 2)
title('Pointwise Error (%)')
xlabel('Error (%)')
ylabel('y position')
grid on

sgtitle(sprintf('LBM Poiseuille Flow | omega=%.2f | nu=%.6f', omega, nu))

saveas(gcf, 'lattice_boltzmann.png')