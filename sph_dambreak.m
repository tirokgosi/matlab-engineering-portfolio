%% PROJECT 22: Smoothed Particle Hydrodynamics - 2D Dam Break
% Simplified SPH dam break simulation
% Shows Lagrangian particle nature of meshfree methods

%--- Parameters ---
g     = 9.81;
dt    = 0.005;
t_end = 3.0;
n_steps = round(t_end/dt);

%--- Initial Particle Layout ---
% Water column on left
dx_p  = 0.05;
x_dam = 0.05:dx_p:0.45;
y_dam = 0.05:dx_p:0.85;
[Xd,Yd] = meshgrid(x_dam,y_dam);
pos = [Xd(:),Yd(:)];
N   = size(pos,1);
vel = zeros(N,2);

%--- Domain ---
x_min=0; x_max=2.0;
y_min=0; y_max=1.5;
e    = 0.6;     % Restitution coefficient

fprintf('--- SPH Dam Break Simulation ---\n')
fprintf('Particles : %d\n', N)
fprintf('Domain    : %.1f x %.1f m\n', x_max, y_max)
fprintf('Time steps: %d\n', n_steps)
fprintf('\nRunning simulation...\n')

%--- Smoothing Parameters ---
h     = 0.15;       % Interaction radius
rho0  = 1000;
m_p   = rho0*dx_p^2;
k_p   = 500;        % Pressure stiffness

figure;
plot_interval = 50;

for step = 1:n_steps
    t = step*dt;

    %--- Pressure forces ---
    acc = zeros(N,2);
    acc(:,2) = -g;

    for i = 1:N
        dx = pos(:,1)-pos(i,1);
        dy = pos(:,2)-pos(i,2);
        r  = sqrt(dx.^2+dy.^2);
        nbr = find(r>1e-4 & r<h);

        for j = nbr'
            rij = r(j);
            % Simple repulsive pressure
            f_mag = k_p*(h-rij)/rij;
            acc(i,1) = acc(i,1) + f_mag*dx(j);
            acc(i,2) = acc(i,2) + f_mag*dy(j);
        end
    end

    %--- Damping ---
    acc = acc - 0.5*vel;

    %--- Integrate ---
    vel = vel + acc*dt;
    pos = pos + vel*dt;

    %--- Boundary Conditions ---
    hit = pos(:,1)<x_min+0.01;
    pos(hit,1) = x_min+0.01;
    vel(hit,1) = abs(vel(hit,1))*e;

    hit = pos(:,1)>x_max-0.01;
    pos(hit,1) = x_max-0.01;
    vel(hit,1) = -abs(vel(hit,1))*e;

    hit = pos(:,2)<y_min+0.01;
    pos(hit,2) = y_min+0.01;
    vel(hit,2) = abs(vel(hit,2))*e;

    hit = pos(:,2)>y_max-0.01;
    pos(hit,2) = y_max-0.01;
    vel(hit,2) = -abs(vel(hit,2))*e;

    %--- Plot ---
    if mod(step,plot_interval)==0
        speed = sqrt(vel(:,1).^2+vel(:,2).^2);
        scatter(pos(:,1),pos(:,2),25,speed,'filled')
        colormap('jet')
        colorbar
        clim([0,3])
        hold on
        % Draw walls
        rectangle('Position',[x_min,y_min, ...
                   x_max-x_min,y_max-y_min], ...
                   'EdgeColor','k','LineWidth',2)
        hold off
        xlabel('x (m)'); ylabel('y (m)')
        title(sprintf('SPH Dam Break | t=%.2f s | %d particles', ...
                       t,N))
        xlim([x_min-0.05, x_max+0.05])
        ylim([y_min-0.05, y_max+0.05])
        axis equal
        drawnow
    end
end

fprintf('Simulation complete\n')
fprintf('Final spread: x=[%.2f, %.2f] m\n', ...
         min(pos(:,1)),max(pos(:,1)))
fprintf('Particles spread across %.1f%% of domain\n', ...
         (max(pos(:,1))-min(pos(:,1)))/x_max*100)

saveas(gcf,'sph_dambreak.png')