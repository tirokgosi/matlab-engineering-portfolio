%% PROJECT 16: Spectral Collocation for a Vibrating Membrane
% Solves d²u/dx² + d²u/dy² = lambda*u for drumhead eigenmodes
% Uses Chebyshev pseudospectral method
% Achieves high accuracy with only 20x20 grid points

%--- Chebyshev Differentiation Matrix ---
% Input:  N = number of interior points
% Output: D = differentiation matrix, x = Chebyshev points
function [D, x] = cheb(N)
if N == 0
    D = 0; x = 1; return
end
x  = cos(pi*(0:N)/N)';          % Chebyshev points on [-1,1]
c  = [2; ones(N-1,1); 2] .* (-1).^(0:N)';
X  = repmat(x, 1, N+1);
dX = X - X';
D  = (c*(1./c)') ./ (dX + eye(N+1));
D  = D - diag(sum(D,2));        % Diagonal correction
end

%--- Grid Setup ---
N  = 20;        % Number of Chebyshev points per direction
[D, x] = cheb(N);

% Second derivative matrix
D2 = D^2;

% Interior points only (remove boundary rows/columns)
% Boundary condition: u = 0 at edges
D2_int = D2(2:N, 2:N);     % Interior second derivative matrix

fprintf('--- Spectral Membrane Solver ---\n')
fprintf('Grid points per direction : %d\n', N)
fprintf('Interior points           : %d x %d\n', N-1, N-1)
fprintf('Chebyshev points computed\n')
fprintf('Second derivative matrix  : %d x %d\n', N-1, N-1)
%--- Assemble 2D Laplacian ---
% For a 2D problem we use Kronecker products to combine
% the 1D operators in x and y directions
% L = D2x (x) I + I (x) D2y
% Since domain is square D2x = D2y = D2_int

I   = eye(N-1);
L2D = kron(I, D2_int) + kron(D2_int, I);

fprintf('\nAssembling 2D Laplacian...\n')
fprintf('Laplacian matrix size: %d x %d\n', size(L2D,1), size(L2D,2))

%--- Solve Eigenvalue Problem ---
% L2D * u = lambda * u
% We want the smallest (most negative) eigenvalues
% These correspond to lowest frequency modes
fprintf('Solving eigenvalue problem...\n')

n_modes = 6;
[V, D_eig] = eigs(L2D, n_modes, 'smallestabs');

% Extract eigenvalues and sort
eigenvalues = diag(D_eig);
[eigenvalues, idx] = sort(eigenvalues, 'descend');
V = V(:, idx);

%--- Analytical Eigenvalues for Square Membrane ---
% lambda_mn = -(m^2 + n^2)*pi^2/4  for unit square [-1,1]x[-1,1]
fprintf('\n--- Eigenvalue Comparison ---\n')
fprintf('Mode | Numerical | Analytical | Error\n')
fprintf('-----|-----------|------------|------\n')
modes_mn = [1,1; 1,2; 2,1; 2,2; 1,3; 3,1];
for i = 1:6
    m   = modes_mn(i,1);
    n   = modes_mn(i,2);
    lam_exact = -(m^2+n^2)*pi^2/4;
    err = abs(eigenvalues(i) - lam_exact)/abs(lam_exact)*100;
    fprintf('%4d | %9.4f | %10.4f | %.2e%%\n', ...
        i, eigenvalues(i), lam_exact, err)
end
%--- Reconstruct Mode Shapes on 2D Grid ---
x_int = x(2:N);         % Interior Chebyshev points
[Xg, Yg] = meshgrid(x_int, x_int);

%--- Plot 6 Mode Shapes ---
figure;

mode_names = {'Mode 1 (1,1)', 'Mode 2 (1,2)', 'Mode 3 (2,1)', ...
    'Mode 4 (2,2)', 'Mode 5 (1,3)', 'Mode 6 (3,1)'};

for i = 1:6

    subplot(2,3,i)

    % Reshape eigenvector to 2D grid
    mode_shape = reshape(V(:,i), N-1, N-1);

    % Normalize mode shape
    mode_shape = mode_shape / max(abs(mode_shape(:)));

    % Plot contour
    contourf(Xg, Yg, mode_shape, 20, 'LineStyle', 'none')
    colormap('jet')
    colorbar

    title(sprintf('%s\n\\lambda = %.4f', ...
        mode_names{i}, eigenvalues(i)))
    xlabel('x')
    ylabel('y')
    axis equal tight

end

sgtitle('Chebyshev Spectral - First 6 Drumhead Vibration Modes')

%--- Print accuracy insight ---
fprintf('\n--- Spectral Method Accuracy ---\n')
fprintf('Grid points used  : %d x %d = %d total\n', N, N, N^2)
fprintf('FDM needs ~200x200 = 40000 points for same accuracy\n')
fprintf('Spectral uses only %d points - %.0fx fewer\n', ...
    N^2, 40000/N^2)
fprintf('This is exponential convergence\n')

%--- Save ---
saveas(gcf, 'spectral_membrane.png')