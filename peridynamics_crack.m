%% PROJECT 23: Peridynamics for Dynamic Crack Propagation
% 2D brittle plate under tensile load
% Bonds break irreversibly when stretch exceeds critical value
% Crack nucleation and propagation emerge naturally

%--- Material Properties ---
E     = 200e9;      % Young's modulus Pa
rho   = 7800;       % Density kg/m^3
Gc    = 100;        % Fracture energy J/m^2

%--- Discretization ---
Lx    = 0.1;        % Plate width m
Ly    = 0.1;        % Plate height m
dx    = 0.005;      % Particle spacing m
delta = 3.015*dx;   % Horizon radius m

%--- Grid of Particles ---
x_pts = dx/2:dx:Lx-dx/2;
y_pts = dx/2:dx:Ly-dx/2;
[Xp,Yp] = meshgrid(x_pts,y_pts);
pos   = [Xp(:),Yp(:)];
N     = size(pos,1);
vel   = zeros(N,2);
acc   = zeros(N,2);

%--- Material Point Volume ---
V     = dx^2;       % 2D volume (area)

%--- Peridynamic Constants ---
% Bond stiffness c for 2D
c     = 9*E/(pi*delta^3);

%--- Critical Stretch ---
s0    = sqrt(4*pi*Gc/(9*E*delta));

%--- Pre-existing Crack ---
% Horizontal crack at mid-height, left half
crack_y  = Ly/2;
crack_x  = Lx/2;

%--- Bond Connectivity ---
fprintf('--- Peridynamics Crack Propagation ---\n')
fprintf('Particles    : %d\n', N)
fprintf('Horizon      : %.4f m\n', delta)
fprintf('Critical stretch: %.4f\n', s0)
fprintf('\nBuilding bond connectivity...\n')

% Find bonds within horizon
bonds      = [];
bond_broken = [];
for i = 1:N
    for j = i+1:N
        dx_ij = pos(j,1)-pos(i,1);
        dy_ij = pos(j,2)-pos(i,2);
        r     = sqrt(dx_ij^2+dy_ij^2);
        if r <= delta
            % Check if bond crosses pre-existing crack
            crosses = false;
            if abs(pos(i,2)-crack_y)<dx && abs(pos(j,2)-crack_y)<dx
                if min(pos(i,1),pos(j,1)) < crack_x
                    crosses = true;
                end
            end
            bonds       = [bonds;      i, j, r];
            bond_broken = [bond_broken; crosses];
        end
    end
end

n_bonds = size(bonds,1);
fprintf('Total bonds  : %d\n', n_bonds)
fprintf('Pre-cracked  : %d bonds\n', sum(bond_broken))

%--- Time Integration ---
cd_wave = sqrt(E/rho);
dt      = 0.5*dx/cd_wave;
t_end   = 50*dt;
n_steps = round(t_end/dt);

%--- Loading: Applied velocity at top and bottom ---
load_vel = 0.5;     % m/s outward

fprintf('Time steps   : %d\n', n_steps)
fprintf('\nRunning peridynamic simulation...\n')

damage   = zeros(N,1);
n_bonds_per_node = zeros(N,1);
for b = 1:n_bonds
    i = bonds(b,1); j = bonds(b,2);
    n_bonds_per_node(i) = n_bonds_per_node(i)+1;
    n_bonds_per_node(j) = n_bonds_per_node(j)+1;
end

figure;

for step = 1:n_steps

    %--- Apply boundary loading ---
    % Top layer moves up, bottom layer moves down
    top_nodes = pos(:,2) > Ly - 2*dx;
    bot_nodes = pos(:,2) < 2*dx;
    vel(top_nodes,2) =  load_vel;
    vel(bot_nodes,2) = -load_vel;

    %--- Compute peridynamic forces ---
    F_pd = zeros(N,2);
    broken_this_step = 0;

    for b = 1:n_bonds
        if bond_broken(b); continue; end

        i   = bonds(b,1);
        j   = bonds(b,2);
        L0  = bonds(b,3);

        % Current bond vector
        xi  = (pos(j,:)+vel(j,:)*dt) - (pos(i,:)+vel(i,:)*dt);
        L   = norm(xi);

        % Bond stretch
        s   = (L-L0)/L0;

        % Check failure
        if abs(s) > s0
            bond_broken(b)   = true;
            broken_this_step = broken_this_step+1;
            continue
        end

        % Bond force
        f_mag = c*s*V*V/L0;
        f_vec = f_mag*(xi/L);

        F_pd(i,:) = F_pd(i,:) + f_vec;
        F_pd(j,:) = F_pd(j,:) - f_vec;
    end

    %--- Update damage ---
    n_broken = zeros(N,1);
    for b = 1:n_bonds
        if bond_broken(b)
            i = bonds(b,1); j = bonds(b,2);
            n_broken(i) = n_broken(i)+1;
            n_broken(j) = n_broken(j)+1;
        end
    end
    damage = n_broken ./ max(n_bonds_per_node,1);

    %--- Velocity Verlet ---
    acc_new = F_pd/rho;
    vel     = vel + 0.5*(acc+acc_new)*dt;
    pos     = pos + vel*dt;
    acc     = acc_new;

    %--- Plot every 10 steps ---
    if mod(step,10)==0
        scatter(pos(:,1)*1000, pos(:,2)*1000, ...
                20, damage, 'filled')
        colormap('jet')
        colorbar
        clim([0,1])
        xlabel('x (mm)'); ylabel('y (mm)')
        title(sprintf('Peridynamic Damage | t=%.2e s | Broken bonds=%d', ...
                       step*dt, sum(bond_broken)))
        axis equal tight
        drawnow
    end

end

fprintf('Simulation complete\n')
fprintf('Total broken bonds : %d / %d\n', sum(bond_broken), n_bonds)
fprintf('Max damage         : %.3f\n', max(damage))
fprintf('Crack propagated   : %s\n', string(max(damage)>0.5))

saveas(gcf,'peridynamics_crack.png')