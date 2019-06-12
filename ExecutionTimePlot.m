load EstimatorRunTime.mat

EstimatorRunTime

plot(10:1:15, EstimatorRunTime(10:1:15,1),'--s',  'LineWidth', 2)
hold on
plot(10:1:15, EstimatorRunTime(10:1:15,2),':+',  'LineWidth', 2)
hold on
plot(10:1:15, EstimatorRunTime(10:1:15,3),'-.^',  'LineWidth', 2)

grid on
xticks(10:1:15)
set(gca, 'FontSize',14, 'FontName', 'Times New Roman')
xlabel('\fontsize{14}\fontname{Times New Roman}Time');
ylabel(['\fontsize{14}\fontname{Times New Roman}Program Execution ' ...
        'Time (s)']);
legend('\fontsize{14}\fontname{Times New Roman}MMSE Estimator', ...
        '\fontsize{14}\fontname{Times New Roman}Fixed Memory Estimator', ...
       '\fontsize{14}\fontname{Times New Roman}Particle Filter');




