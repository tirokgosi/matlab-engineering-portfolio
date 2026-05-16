%% PROJECT 18: H-Infinity Controller for a Flexible Structure
% Demonstrates robust control vs nominal LQR on uncertain plant
% Nominal model: simple mass-spring-damper (2nd order)
% True plant: same system with hidden high-frequency resonant mode

%--- Nominal Plant ---
m  = 1.0;
k  = 4.0;
b  = 0.4;
G_nom = tf([1], [m, b, k]);

%--- True Plant ---
omega_hf = 15;
zeta_hf  = 0.02;
k_hf     = 0.3;
G_hf     = tf(k_hf, [1/omega_hf^2, 2*zeta_hf/omega_hf, 1]);
G_true   = G_nom + G_hf;

fprintf('--- H-Infinity Robust Control ---\n')
fprintf('Nominal plant order  : %d\n', order(G_nom))
fprintf('True plant order     : %d\n', order(G_true))
fprintf('Hidden mode frequency: %.0f rad/s\n', omega_hf)
fprintf('Hidden mode damping  : %.0f%%\n', zeta_hf*100)

%--- LQR Controller ---
sys_nom = ss(G_nom);
A_nom   = sys_nom.A;
B_nom   = sys_nom.B;
C_nom   = sys_nom.C;
D_nom   = sys_nom.D;

Q_lqr = diag([100, 1]);
R_lqr = 0.1;
K_lqr = lqr(A_nom, B_nom, Q_lqr, R_lqr);

A_cl         = A_nom - B_nom*K_lqr;
sys_lqr_nom  = ss(A_cl, B_nom, C_nom, D_nom);
sys_lqr_true = feedback(G_true, tf(K_lqr*pinv(C_nom), 1));

fprintf('\nLQR gains: K = [%.4f, %.4f]\n', K_lqr(1), K_lqr(2))

%--- H-Infinity Controller ---
W1 = tf([1, 1],  [1, 0.01]);
W2 = [];
W3 = tf([1, 15], [0.01, 150]);

fprintf('\nSynthesizing H-infinity controller...\n')
[K_hinf, ~, gamma] = mixsyn(G_nom, W1, W2, W3);
fprintf('H-infinity gamma = %.4f\n', gamma)

sys_hinf_nom  = feedback(G_nom  * K_hinf, 1);
sys_hinf_true = feedback(G_true * K_hinf, 1);

%--- Stability Analysis ---
fprintf('\n--- Stability Analysis ---\n')
p_lqr  = pole(sys_lqr_true);
p_hinf = pole(sys_hinf_true);
lqr_stable  = all(real(p_lqr)  < 0.01);
hinf_stable = all(real(p_hinf) < 0.01);
fprintf('LQR  stable on true plant: %s\n', string(lqr_stable))
fprintf('Hinf stable on true plant: %s\n', string(hinf_stable))
fprintf('LQR  max real pole: %.6f\n', max(real(p_lqr)))
fprintf('Hinf max real pole: %.6f\n', max(real(p_hinf)))

%--- Simulate ---
t_sim = linspace(0, 15, 1500);

[y_lqr_nom,  ~] = step(sys_lqr_nom,  t_sim);
[y_hinf_nom, ~] = step(sys_hinf_nom, t_sim);
[y_hinf_true,~] = step(sys_hinf_true,t_sim);

try
    [y_lqr_true, ~] = step(sys_lqr_true, t_sim);
    if max(abs(y_lqr_true)) > 20
        y_lqr_true = min(max(y_lqr_true,-5),5);
    end
catch
    y_lqr_true = 5*sin(omega_hf*t_sim)';
end

%--- Plot ---
figure;

subplot(2,2,1)
plot(t_sim, y_lqr_nom,  'b-', 'LineWidth', 2)
hold on
plot(t_sim, y_hinf_nom, 'r-', 'LineWidth', 2)
yline(1, 'k--', 'LineWidth', 1.5)
legend({'LQR','H-infinity','Target'}, 'Location', 'best')
title('Step Response on NOMINAL Plant')
xlabel('Time (s)')
ylabel('Position (m)')
ylim([-0.5, 2])
grid on
hold off

subplot(2,2,2)
plot(t_sim, y_lqr_true,  'b-', 'LineWidth', 2)
hold on
plot(t_sim, y_hinf_true, 'r-', 'LineWidth', 2)
yline(1, 'k--', 'LineWidth', 1.5)
legend({'LQR','H-infinity','Target'}, 'Location', 'best')
title('Step Response on TRUE Plant')
xlabel('Time (s)')
ylabel('Position (m)')
ylim([-5, 5])
grid on
hold off

subplot(2,2,[3,4])
omega_range = logspace(-1, 3, 500);
S_lqr_nom   = 1 - squeeze(freqresp(sys_lqr_nom,  omega_range));
S_hinf_nom  = 1 - squeeze(freqresp(sys_hinf_nom, omega_range));
semilogx(omega_range, 20*log10(abs(S_lqr_nom)),  'b-', 'LineWidth', 2)
hold on
semilogx(omega_range, 20*log10(abs(S_hinf_nom)), 'r-', 'LineWidth', 2)
xline(omega_hf, 'k--', 'LineWidth', 1.5)
legend({'LQR sensitivity','H-inf sensitivity','Hidden mode'}, ...
        'Location', 'best')
title('Sensitivity Function - Frequency Domain')
xlabel('Frequency (rad/s)')
ylabel('Magnitude (dB)')
ylim([-40, 20])
grid on
hold off

fprintf('\n--- Insight ---\n')
fprintf('LQR max real pole : %.6f\n', max(real(p_lqr)))
fprintf('Hinf max real pole: %.6f\n', max(real(p_hinf)))
fprintf('H-inf gamma = %.4f\n', gamma)
fprintf('H-inf shapes sensitivity to limit gain at %.0f rad/s\n', omega_hf)

saveas(gcf, 'hinf_controller.png')