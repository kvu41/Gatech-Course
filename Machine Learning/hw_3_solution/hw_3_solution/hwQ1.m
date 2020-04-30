% hw3 Q1 self study

clear; close all;

rawdata = readtable('n90pol.csv');

data = table2array(rawdata);

m = size(data, 1);

%%
idx_all = unique(data(:,3));
nclass = length(idx_all);
idx_single = cell(nclass, 1);
data_single = cell(nclass, 1);
for ii = 1:nclass
    idx = idx_all(ii);
    idx_single{ii} = find(data(:, 3) == idx);
    data_single{ii} = data(idx_single{ii},1:2);
end

%% histogram
nbin = 20;
edges = linspace(-0.1, 0.1, nbin);
figure;
hist3(data_single{1}(:,1:2), {edges,edges}, 'FaceColor', [215,25,28]/255, 'FaceAlpha',0.5); hold on
hist3(data_single{2}(:,1:2), {edges,edges}, 'FaceColor', [253,174,97]/255, 'FaceAlpha',0.5); hold on
hist3(data_single{3}(:,1:2), {edges,edges}, 'FaceColor', [171,217,233]/255, 'FaceAlpha',0.5); hold on
hist3(data_single{4}(:,1:2), {edges,edges}, 'FaceColor', [44,123,182]/255, 'FaceAlpha',0.5)
axis square
legend('orientation 2','orientation 3','orientation 4','orientation 5' )

%% KDE

gridx1 = -0.1:.005:.1;
gridx2 = -0.1:.005:.1;
[x1,x2] = meshgrid(gridx1, gridx2);
x1 = x1(:);
x2 = x2(:);
xi = [x1 x2];

figure;
for ii = 1:4
    subplot(2, 2, ii)
    ksdensity(data_single{ii}, xi, 'PlotFcn','contour');
    title(['orientation: ', num2str(ii+1)]);
    xlabel('amygdala')
    ylabel('gcc')
    axis square
    grid on
end
suptitle('KDE, countor plot for each class')

%% conditional probability
% prior
pr = zeros(1, 4);
for ii = 1:4
    pr(ii) = size(data_single{ii}, 1)/size(data, 1);
end

% conditional on amygdala
p_condi_amygdala = zeros(4, m);
figure;
pts = sort(data(:, 1));
for ii = 1:4
    [f,~] = ksdensity(data_single{ii}(:, 1), pts);
    p_condi_amygdala(ii, :) = f*pr(ii);
    plot(pts, p_condi_amygdala(ii, :)); hold on
end
legend('orientation 2','orientation 3','orientation 4','orientation 5' )
title('conditional probability of amygdala')
xlabel('amygdala'); ylabel('conditional probability')

% conditional on amygdala
p_condi_gcc = zeros(4, m);
figure;
pts = sort(data(:, 2));
for ii = 1:4
    [f,~] = ksdensity(data_single{ii}(:, 2), pts);
    p_condi_gcc(ii, :) = f*pr(ii);
    plot(pts, p_condi_gcc(ii, :)); hold on
end
legend('orientation 2','orientation 3','orientation 4','orientation 5' )
title('conditional probability of gcc')
xlabel('gcc'); ylabel('conditional probability')


