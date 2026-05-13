%% PROJECT 4: 4-Bar Linkage Analysis & Animation
% Solves position of a 4-bar linkage using Newton-Raphson method
% Animates the full crank rotation and plots transmission angle

%--- Link Lengths (meters) ---
L1 = 2.0;     % Ground link (fixed)
L2 = 1.0;     % Crank (input - we rotate this)
L3 = 2.5;     % Coupler (connects crank to follower)
L4 = 2.0;     % Follower (output)

%--- Check Grashof Condition ---
% Grashof condition: shortest + longest < sum of other two
% This guarantees the crank can rotate fully (360 degrees)
links = sort([L1, L2, L3, L4]);
if (links(1) + links(4)) < (links(2) + links(3))
    fprintf('Grashof condition satisfied - crank can rotate fully\n')
else
    fprintf('Warning: Grashof condition NOT satisfied\n')
end

%--- Crank Angles to Simulate ---
theta2 = linspace(0, 2*pi, 360);   % Full 360 degree rotation, 1 degree steps

%--- Storage for results ---
theta3 = zeros(1, 360);    % Coupler angles
theta4 = zeros(1, 360);    % Follower angles

%--- Newton-Raphson Solver ---
% Initial guess for first position
t3 = pi/4;      % Initial guess for coupler angle
t4 = pi/6;      % Initial guess for follower angle

for i = 1:360

    % Current crank angle
    t2 = theta2(i);

    % Newton-Raphson iteration
    for iter = 1:100    % Maximum 100 iterations

        % Vector loop equations F1 and F2
        % These must equal zero when t3 and t4 are correct
        F1 = L2*cos(t2) + L3*cos(t3) - L4*cos(t4) - L1;
        F2 = L2*sin(t2) + L3*sin(t3) - L4*sin(t4);

        % Check if we have converged (solution is close enough)
        if sqrt(F1^2 + F2^2) < 1e-10
            break
        end

        % Jacobian matrix (derivatives of F1 and F2)
        J = [-L3*sin(t3),  L4*sin(t4);
            L3*cos(t3), -L4*cos(t4)];

        % Newton-Raphson update step
        delta = J \ [-F1; -F2];
        t3 = t3 + delta(1);
        t4 = t4 + delta(2);

    end

    % Store converged angles
    theta3(i) = t3;
    theta4(i) = t4;

    % Use current solution as initial guess for next angle
    % This makes convergence fast and reliable

end

fprintf('Newton-Raphson solver complete\n')

%--- Calculate Transmission Angle ---
% Transmission angle = angle between coupler and follower
% Best when close to 90 degrees, bad when close to 0 or 180
gamma = abs(theta3 - theta4);

%--- Animation ---
figure;
subplot(1,2,1)      % Left plot: linkage animation
title('4-Bar Linkage Animation')
xlabel('x (m)')
ylabel('y (m)')
axis([-2, 4, -2, 3])
grid on
hold on

% Fixed pivot points
O2 = [0, 0];        % Crank pivot (origin)
O4 = [L1, 0];       % Follower pivot

for i = 1:5:360     % Animate every 5th frame for speed

    cla             % Clear previous frame

    t2 = theta2(i);
    t3 = theta3(i);
    t4 = theta4(i);

    % Calculate joint positions
    A = O2 + L2*[cos(t2), sin(t2)];          % Crank end
    B = O4 + L4*[cos(t4), sin(t4)];          % Follower end

    % Draw ground link
    plot([O2(1), O4(1)], [O2(2), O4(2)], ...
        'k-', 'LineWidth', 3)

    % Draw crank
    plot([O2(1), A(1)], [O2(2), A(2)], ...
        'b-o', 'LineWidth', 2)

    % Draw coupler
    plot([A(1), B(1)], [A(2), B(2)], ...
        'g-o', 'LineWidth', 2)

    % Draw follower
    plot([O4(1), B(1)], [O4(2), B(2)], ...
        'r-o', 'LineWidth', 2)

    % Draw fixed supports
    plot(O2(1), O2(2), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k')
    plot(O4(1), O4(2), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k')

    legend('Ground','Crank','Coupler','Follower')
    drawnow         % Update the figure immediately

end

%--- Transmission Angle Plot ---
subplot(1,2,2)      % Right plot: transmission angle
plot(rad2deg(theta2), rad2deg(gamma), 'm-', 'LineWidth', 2)
yline(90, 'k--', 'Ideal 90 degrees', 'LineWidth', 1.5)
yline(40, 'r--', 'Lower Limit', 'LineWidth', 1.5)
xlabel('Crank Angle (degrees)')
ylabel('Transmission Angle (degrees)')
title('Transmission Angle over Full Crank Rotation')
grid on

%--- Save figure ---
saveas(gcf, 'fourbar_linkage.png')