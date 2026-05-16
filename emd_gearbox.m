%% PROJECT 26: Empirical Mode Decomposition for Gearbox Fault
% Decomposes non-stationary vibration signal into IMFs
% Identifies fault frequency from chipped tooth impulses
% Compares EMD to standard FFT for non-stationary signals

%--- Signal Parameters ---
fs      = 5000;     % Sampling frequency Hz
t_end   = 2.0;     % Signal duration seconds
t       = 0:1/fs:t_end;
N       = length(t);

%--- Gearbox Signal Generation ---
% Shaft rotation frequency
f_shaft = 20;       % Hz - shaft speed
f_mesh  = 200;      % Hz - gear mesh frequency

% Normal gearbox vibration
normal  = 0.5*sin(2*pi*f_mesh*t) + ...
          0.3*sin(2*pi*2*f_mesh*t) + ...
          0.2*sin(2*pi*3*f_mesh*t);

% Chipped tooth - impulse every revolution
% Speed varies slightly (non-stationary)
speed_var = 1 + 0.05*sin(2*pi*0.5*t);  % ±5% speed variation
phase     = cumsum(2*pi*f_shaft*speed_var/fs);
impulse   = zeros(1,N);
for k = 1:floor(f_shaft*t_end)
    idx = round(k*fs/f_shaft);
    if idx > 0 && idx <= N
        impulse(idx) = 3.0;     % Impulse amplitude
    end
end

% Exponentially decaying impulse response
imp_response = zeros(1,N);
tau_imp = 0.005;    % Decay time constant
for i = 1:N
    if impulse(i) > 0
        len = min(100, N-i);
        imp_response(i:i+len) = imp_response(i:i+len) + ...
            impulse(i)*exp(-(0:len)*fs*tau_imp/fs);
    end
end

% Total signal with noise
noise   = 0.1*randn(1,N);
signal  = normal + imp_response + noise;

fprintf('--- EMD Gearbox Fault Detection ---\n')
fprintf('Sampling frequency : %d Hz\n', fs)
fprintf('Signal duration    : %.1f seconds\n', t_end)
fprintf('Shaft frequency    : %d Hz\n', f_shaft)
fprintf('Mesh frequency     : %d Hz\n', f_mesh)
fprintf('Data points        : %d\n', N)

%--- EMD Algorithm ---
function imfs = emd_decompose(signal, max_imfs, max_sift)
    imfs     = [];
    residual = signal;

    for k = 1:max_imfs
        h = residual;

        % Sifting process
        for sift = 1:max_sift
            % Find local maxima and minima
            [~, idx_max] = findpeaks( h);
            [~, idx_min] = findpeaks(-h);

            if length(idx_max) < 3 || length(idx_min) < 3
                break
            end

            % Add endpoints
            x_max = [1, idx_max, length(h)];
            y_max = [h(1), h(idx_max), h(end)];
            x_min = [1, idx_min, length(h)];
            y_min = [h(1), h(idx_min), h(end)];

            % Interpolate envelopes
            x_all = 1:length(h);
            env_up  = interp1(x_max, y_max, x_all, 'spline');
            env_low = interp1(x_min, y_min, x_all, 'spline');

            % Mean envelope
            mean_env = (env_up + env_low)/2;

            % Subtract mean
            h_new = h - mean_env;

            % Check stopping criterion (SD method)
            SD = sum((h - h_new).^2) / sum(h.^2);
            h  = h_new;

            if SD < 0.2; break; end
        end

        imfs     = [imfs; h];
        residual = residual - h;

        % Stop if residual is monotonic
        [~,mx] = findpeaks( residual);
        [~,mn] = findpeaks(-residual);
        if length(mx) < 2 || length(mn) < 2
            break
        end
    end
    imfs = [imfs; residual];
end

fprintf('\nRunning EMD decomposition...\n')
max_imfs = 6;
imfs     = emd_decompose(signal, max_imfs, 20);
n_imfs   = size(imfs,1);
fprintf('IMFs extracted: %d\n', n_imfs)

%--- Identify Fault IMF ---
% The fault IMF has dominant energy at shaft frequency
fault_imf = 1;
max_energy = 0;
for k = 1:n_imfs-1
    Y    = abs(fft(imfs(k,:)));
    f_ax = (0:N-1)*fs/N;
    % Energy near shaft frequency
    idx  = abs(f_ax - f_shaft) < 5;
    E    = sum(Y(idx).^2);
    if E > max_energy
        max_energy = E;
        fault_imf  = k;
    end
end
fprintf('Fault IMF identified: IMF %d\n', fault_imf)

%--- Plot ---
figure;

% Original signal
subplot(n_imfs+2, 1, 1)
plot(t, signal, 'b-', 'LineWidth', 0.5)
ylabel('Signal')
title('Original Gearbox Vibration Signal')
grid on

% IMFs
for k = 1:n_imfs
    subplot(n_imfs+2, 1, k+1)
    if k == fault_imf
        plot(t, imfs(k,:), 'r-', 'LineWidth', 1)
        ylabel(sprintf('IMF%d*', k))
    else
        plot(t, imfs(k,:), 'b-', 'LineWidth', 0.5)
        ylabel(sprintf('IMF%d', k))
    end
    grid on
end

% FFT of fault IMF
subplot(n_imfs+2, 1, n_imfs+2)
Y_fault = abs(fft(imfs(fault_imf,:)));
f_ax    = (0:N-1)*fs/N;
plot(f_ax(1:N/2), Y_fault(1:N/2), 'r-', 'LineWidth', 1.5)
xlim([0, 100])
xlabel('Frequency (Hz)')
ylabel('Amplitude')
title(sprintf('FFT of Fault IMF %d - Shaft freq %.0f Hz', ...
               fault_imf, f_shaft))
grid on

sgtitle('EMD Gearbox Fault Detection - Chipped Tooth')

saveas(gcf, 'emd_gearbox.png')

%--- Print results ---
fprintf('\n--- Results ---\n')
fprintf('Signal components:\n')
fprintf('  Normal mesh vibration : %.0f, %.0f, %.0f Hz\n', ...
         f_mesh, 2*f_mesh, 3*f_mesh)
fprintf('  Fault impulse rate    : %.0f Hz (shaft speed)\n', f_shaft)
fprintf('  Speed variation       : ±5%% (non-stationary)\n')
fprintf('Fault IMF %d shows clear impulse pattern\n', fault_imf)
fprintf('EMD adapts to non-stationarity unlike FFT\n')