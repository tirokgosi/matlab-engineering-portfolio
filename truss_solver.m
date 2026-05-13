%% PROJECT 1: 2D Truss Solver
% We define the geometry, material, and loading of a simple 3-node truss

%--- Node Coordinates [x, y] in meters ---
% Row 1 = Node 1, Row 2 = Node 2, Row 3 = Node 3
nodes = [0, 0;       % Node 1: bottom left (fixed support)
    1, 1;       % Node 2: top middle
    2, 0];      % Node 3: bottom right (roller support)

%--- Element Connectivity ---
% Each row = one bar. [start node, end node]
elements = [1, 2;    % Bar 1: connects Node1 to Node2
    2, 3;    % Bar 2: connects Node2 to Node3
    1, 3];   % Bar 3: connects Node1 to Node3

%--- Material Properties ---
E = 200e9;   % Young's Modulus in Pascals (steel)
A = 0.01;    % Cross-sectional area in m²

%--- Force Vector ---
% Each node has 2 degrees of freedom: x-direction and y-direction
% So for 3 nodes we have 6 entries total: [F1x, F1y, F2x, F2y, F3x, F3y]
F = zeros(6, 1);     % Start with all zeros (no load anywhere)
F(4) = -50000;       % Apply 50,000 N downward at Node 2 (entry 4 = Node2 y-direction)

%--- Global Stiffness Matrix ---
% 6x6 matrix of zeros that we will fill in bar by bar
K_global = zeros(6, 6);

%--- Assemble Global Stiffness Matrix ---
for i = 1:3      % Loop through all 3 bars

    % Get the start and end node numbers for this bar
    n1 = elements(i, 1);    % Start node
    n2 = elements(i, 2);    % End node

    % Get the x,y coordinates of each node
    x1 = nodes(n1, 1);   y1 = nodes(n1, 2);
    x2 = nodes(n2, 1);   y2 = nodes(n2, 2);

    % Calculate bar length
    L = sqrt((x2-x1)^2 + (y2-y1)^2);

    % Calculate angle of the bar
    c = (x2-x1)/L;    % cosine of angle
    s = (y2-y1)/L;    % sine of angle

    % Local stiffness matrix for this bar (4x4)
    k = (E*A/L) * [c*c,  c*s, -c*c, -c*s;
        c*s,  s*s, -c*s, -s*s;
        -c*c, -c*s,  c*c,  c*s;
        -c*s, -s*s,  c*s,  s*s];

    % Find which global entries this bar maps to
    dofs = [2*n1-1, 2*n1, 2*n2-1, 2*n2];

    % Add this bar's stiffness into the global matrix
    K_global(dofs, dofs) = K_global(dofs, dofs) + k;

end

%--- Apply Boundary Conditions ---
% Node 1 is FIXED: cannot move in x or y (entries 1 and 2)
% Node 3 is a ROLLER: cannot move in y only (entry 6)
% We use the penalty method: make those entries extremely stiff

penalty = 1e20;   % A very large number

K_global(1,1) = K_global(1,1) + penalty;   % Node 1 x-direction fixed
K_global(2,2) = K_global(2,2) + penalty;   % Node 1 y-direction fixed
K_global(6,6) = K_global(6,6) + penalty;   % Node 3 y-direction fixed

%--- Solve for Displacements ---
U = K_global \ F;

%--- Print Results ---
fprintf('\n--- Nodal Displacements ---\n')
fprintf('Node 1: x = %.6f m,  y = %.6f m\n', U(1), U(2))
fprintf('Node 2: x = %.6f m,  y = %.6f m\n', U(3), U(4))
fprintf('Node 3: x = %.6f m,  y = %.6f m\n', U(5), U(6))

%--- Plot Undeformed and Deformed Shape ---
scale = 1000;    % Exaggerate displacement so we can actually see it

% Deformed node positions
nodes_def = nodes + scale * [U(1),U(2); U(3),U(4); U(5),U(6)];

figure;
hold on;

for i = 1:3
    n1 = elements(i,1);
    n2 = elements(i,2);

    % Original shape in blue
    plot([nodes(n1,1), nodes(n2,1)], ...
        [nodes(n1,2), nodes(n2,2)], ...
        'b-o', 'LineWidth', 2)

    % Deformed shape in red
    plot([nodes_def(n1,1), nodes_def(n2,1)], ...
        [nodes_def(n1,2), nodes_def(n2,2)], ...
        'r--o', 'LineWidth', 2)
end

legend('Undeformed', 'Deformed')
title('2D Truss - Undeformed vs Deformed Shape')
xlabel('x (m)')
ylabel('y (m)')
grid on
hold off
saveas(gcf, 'truss_deformed.png')