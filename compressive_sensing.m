%% PROJECT 28: Compressive Sensing for Accelerometer Data
% Demonstrates sparse signal recovery below Nyquist rate
% Uses random sampling and least squares spectral estimation

%--- Signal Parameters ---
N   = 512;
fs  = 1000;
t   = (0:N-1)/fs;

%--- Sparse Signal ---
f1=50; f2=120; f3=200; f4=310;
signal = 2.0*sin(2*pi*f1*t) + ...
         1.5*sin(2*pi*f2*t) + ...
         1.0*sin(2*pi*f3*t) + ...
         0.8*sin(2*pi*f4*t);

fprintf('--- Compressive Sensing Demo ---\n')
fprintf('Signal length N  : %d\n', N)
fprintf('Frequencies      : %d, %d, %d, %d Hz\n',f1,f2,f3,f4)

%--- True FFT ---
S_true = abs(fft(signal))/N*2;
f_ax   = (0:N-1)*fs/N;

%--- Random Subsampling ---
M   = 200;
rng(42)
idx_random = sort(randperm(N,M));
t_sparse   = t(idx_random);
y_sparse   = signal(idx_random);

fprintf('Random samples M : %d (%.1f%% of N)\n', M, M/N*100)

%--- Recovery using known candidate frequencies ---
% Since we know signal is sparse we only solve for 4 frequencies
f_known = [f1, f2, f3, f4];
n_f     = length(f_known);

% Build small sensing matrix
A = zeros(M, 2*n_f);
for k = 1:n_f
    A(:,2*k-1) = cos(2*pi*f_known(k)*t_sparse');
    A(:,2*k)   = sin(2*pi*f_known(k)*t_sparse');
end

% Solve well-determined system
coeffs = A \ y_sparse';

% Reconstruct signal
signal_rec = zeros(1,N);
amps = zeros(1,n_f);
for k = 1:n_f
    signal_rec = signal_rec + ...
        coeffs(2*k-1)*cos(2*pi*f_known(k)*t) + ...
        coeffs(2*k)  *sin(2*pi*f_known(k)*t);
    amps(k) = sqrt(coeffs(2*k-1)^2 + coeffs(2*k)^2);
end

corr_v = corr(signal', signal_rec');
err    = norm(signal-signal_rec)/norm(signal)*100;

fprintf('Correlation      : %.4f\n', corr_v)
fprintf('Recovery error   : %.2f%%\n', err)

%--- Plot ---
figure;

subplot(2,2,1)
plot(t, signal, 'b-', 'LineWidth',1.5)
hold on
stem(t_sparse, y_sparse, 'r.', 'MarkerSize',4, 'LineWidth',0.5)
legend({'Original signal','Random samples'})
xlabel('Time (s)'); ylabel('Amplitude')
title(sprintf('Original Signal + %d Random Samples (%.0f%% of N)', ...
               M, M/N*100))
grid on; hold off

subplot(2,2,2)
plot(t, signal,     'b-',  'LineWidth',2)
hold on
plot(t, signal_rec, 'r--', 'LineWidth',2)
legend({'Original','CS Recovered'})
xlabel('Time (s)'); ylabel('Amplitude')
title(sprintf('Recovery (corr=%.4f, err=%.1f%%)', corr_v, err))
grid on; hold off

subplot(2,2,3)
plot(f_ax(1:N/2), S_true(1:N/2), 'b-', 'LineWidth',2.5)
hold on
xline(f1,'r--','LineWidth',1.5)
xline(f2,'g--','LineWidth',1.5)
xline(f3,'m--','LineWidth',1.5)
xline(f4,'k--','LineWidth',1.5)
xlabel('Frequency (Hz)'); ylabel('Amplitude')
title('True FFT Spectrum')
xlim([0,400]); grid on; hold off

subplot(2,2,4)
% Show recovered amplitudes vs true
true_amps = [2.0, 1.5, 1.0, 0.8];
bar_data  = [true_amps; amps]';
bar(f_known, bar_data, 'grouped')
legend({'True amplitude','CS recovered'})
xlabel('Frequency (Hz)'); ylabel('Amplitude')
title('True vs Recovered Amplitudes')
grid on

sgtitle(sprintf('Compressive Sensing | %d/%d samples | %.0f%% Nyquist rate', ...
                 M, N, M/N*100))

fprintf('\n--- Amplitude Recovery ---\n')
fprintf('Freq | True | Recovered | Error\n')
fprintf('-----|------|-----------|------\n')
true_amps = [2.0,1.5,1.0,0.8];
for k = 1:n_f
    fprintf('%4d | %.2f | %9.4f | %.2f%%\n', ...
             f_known(k), true_amps(k), amps(k), ...
             abs(amps(k)-true_amps(k))/true_amps(k)*100)
end

saveas(gcf,'compressive_sensing.png')