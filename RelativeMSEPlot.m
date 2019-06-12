load EmpiricalRelativeMSEdata10_1500.mat

EmpiricalRelativeMSE

plot(0:0.1:1, EmpiricalRelativeMSE(:,1), '-o',  'LineWidth', 2)
hold on
plot(0:0.1:1, EmpiricalRelativeMSE(:,2), '--s',  'LineWidth', 2)
hold on
plot(0:0.1:1, EmpiricalRelativeMSE(:,3), ':+',  'LineWidth', 2)
hold on
plot(0:0.1:1, EmpiricalRelativeMSE(:,4), '-.^',  'LineWidth', 2)

grid on
set(gca, 'FontSize',14, 'FontName', 'Times New Roman')
xlabel('\fontsize{14}\fontname{Times New Roman}Packet Loss Rate \it{p}');
ylabel(['\fontsize{14}\fontname{Times New Roman}Relative Sum of MSE']);
legend('\fontsize{14}\fontname{Times New Roman}OLSET-KF', ...
       '\fontsize{14}\fontname{Times New Roman}MMSE Estimator', ...
       '\fontsize{14}\fontname{Times New Roman}Fixed Memory Estimator', ...
       '\fontsize{14}\fontname{Times New Roman}Particle Filter');


