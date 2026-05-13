%% PROJECT 6: Bayesian Inference for Beam Stiffness
% Uses Metropolis MCMC to infer Young's Modulus E
% from noisy cantilever beam deflection measurements

%--- Beam Properties ---
L = 1.0;          % Beam length in meters
b = 0.05;         % Width in meters
h = 0.01;         % Height in meters
I = (b*h^3)/12;   % Second moment of area in m^4
P = 100;          % Applied load in Newtons

%--- True Young's Modulus (what we are trying to find) ---
E_true = 200e9;   % 200 GPa - true value for steel

%--- Cantilever Beam Deflection Formula ---
% delta = PL^3 / (3EI)
% Rearranged: E = PL^3 / (3*I*delta)
deflection_model = @(E) (P * L^3) / (3 * E * I);

%--- Generate 5 Noisy Measurements ---
rng(42)           % Fix random seed for reproducibility
noise_std = 1e-5; % Standard deviation of measurement noise in meters
delta_true = deflection_model(E_true);
measurements = delta_true + noise_std * randn(1, 5);

%--- Print measurements ---
fprintf('\n--- Beam Properties ---\n')
fprintf('True E          = %.1f GPa\n', E_true/1e9)
fprintf('True deflection = %.6f m\n', delta_true)
fprintf('\n--- Noisy Measurements ---\n')
for i = 1:5
    fprintf('Measurement %d: %.6f m\n', i, measurements(i))
end

%--- Bayesian Inference Setup ---
% Bayes theorem: posterior = likelihood * prior / evidence
% We sample the posterior using Metropolis MCMC

%--- Likelihood Function ---
% Probability of observing our measurements given a value of E
% Assumes Gaussian measurement noise
log_likelihood = @(E) sum(-0.5 * ((measurements - deflection_model(E)) ...
    / noise_std).^2) - ...
    length(measurements) * log(noise_std * sqrt(2*pi));

%--- Prior Distribution ---
% What we believed about E before any measurements
% Normal distribution centered at 200 GPa with std of 20 GPa
E_prior_mean = 200e9;     % Prior mean
E_prior_std  = 20e9;      % Prior standard deviation - fairly wide

log_prior = @(E) -0.5 * ((E - E_prior_mean) / E_prior_std)^2 ...
    - log(E_prior_std * sqrt(2*pi));

%--- Log Posterior ---
% Log of likelihood times prior
% We use log to avoid numerical underflow with tiny probabilities
log_posterior = @(E) log_likelihood(E) + log_prior(E);

fprintf('\n--- Bayesian Setup ---\n')
fprintf('Prior mean = %.1f GPa\n', E_prior_mean/1e9)
fprintf('Prior std  = %.1f GPa\n', E_prior_std/1e9)
fprintf('Noise std  = %.2e m\n', noise_std)

%--- Metropolis MCMC Sampler ---
% Algorithm:
% 1. Start at an initial guess for E
% 2. Propose a new E nearby
% 3. Calculate acceptance ratio
% 4. Accept or reject the proposal
% 5. Repeat 10,000 times
% 6. The collection of accepted samples = posterior distribution

%--- MCMC Settings ---
n_samples   = 10000;        % Number of samples to collect
proposal_std = 0.1e9;         % Step size for proposals (0.1 GPa)
E_current   = 180e9;        % Starting guess (deliberately wrong)

%--- Storage ---
samples = zeros(1, n_samples);
accepted = 0;

%--- Run the Sampler ---
fprintf('\nRunning Metropolis MCMC...\n')

for i = 1:n_samples

    % Propose a new E value near current one
    E_proposed = E_current + proposal_std * randn;

    % Only consider positive E values (physical constraint)
    if E_proposed > 0

        % Calculate acceptance ratio in log space
        log_alpha = log_posterior(E_proposed) - log_posterior(E_current);

        % Accept or reject
        if log(rand) < log_alpha
            E_current = E_proposed;     % Accept proposal
            accepted = accepted + 1;
        end

    end

    % Store current sample
    samples(i) = E_current;

end

fprintf('Acceptance rate = %.1f%%\n', 100*accepted/n_samples)
fprintf('Should be between 20%% and 50%% for good mixing\n')

%--- Discard Burn-in Period ---
% First 1000 samples are while algorithm is finding the distribution
% We discard them to get a clean posterior
burnin = 1000;
samples_clean = samples(burnin+1:end);

%--- Calculate 95% Credible Interval ---
CI_low  = prctile(samples_clean, 2.5);    % 2.5th percentile
CI_high = prctile(samples_clean, 97.5);   % 97.5th percentile
E_mean  = mean(samples_clean);
E_std   = std(samples_clean);

%--- Print Results ---
fprintf('\n--- Posterior Results ---\n')
fprintf('True E          = %.4f GPa\n', E_true/1e9)
fprintf('Estimated E     = %.4f GPa\n', E_mean/1e9)
fprintf('Std deviation   = %.4f GPa\n', E_std/1e9)
fprintf('95%% Credible Interval: [%.4f, %.4f] GPa\n', ...
    CI_low/1e9, CI_high/1e9)
fprintf('We are 95%% confident the true E lies in this interval\n')

%--- Plot ---
figure;

%--- Top plot: Markov Chain Trace ---
subplot(2,1,1)
plot(samples/1e9, 'b-', 'LineWidth', 0.5)
yline(E_true/1e9, 'r--', 'True E', 'LineWidth', 2)
yline(E_mean/1e9, 'g--', 'Estimated E', 'LineWidth', 2)
xline(burnin, 'k--', 'End of Burnin', 'LineWidth', 1.5)
xlabel('MCMC Sample Number')
ylabel('E (GPa)')
title('Markov Chain Trace of Young''s Modulus')
grid on

%--- Bottom plot: Posterior Histogram ---
subplot(2,1,2)
histogram(samples_clean/1e9, 60, 'FaceColor', 'b', ...
    'EdgeColor', 'none', 'Normalization', 'probability')
xline(E_true/1e9,  'r--', 'True E',  'LineWidth', 2)
xline(CI_low/1e9,  'k--', '95% CI',  'LineWidth', 2)
xline(CI_high/1e9, 'k--', '95% CI',  'LineWidth', 2)
xlabel('Young''s Modulus E (GPa)')
ylabel('Probability')
title('Posterior Distribution of Young''s Modulus')
grid on

%--- Save figure ---
saveas(gcf, 'bayesian_beam.png')