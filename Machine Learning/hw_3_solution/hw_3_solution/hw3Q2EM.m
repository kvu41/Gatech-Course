% hw3 Q2 EM
clear; close all;
load data
load label
num2 = data(:, find(trueLabel==2))';
num6 = data(:, find(trueLabel==6))';

[N, M] = size(data);

data = data';

%% initializing

prior_old = rand(1, 2);   % mixture prior
prior_old = prior_old/sum(prior_old);

mu1_old = mean(num2); 
mu2_old = mean(num6);
sigma1_old = eye(N); 
sigma2_old = eye(N);

k = 100; % truncation parameter in low rank approximation for inverse Covariance matrix
U = zeros(M, 2);

%% likelihood for initial parameters.
% to avoid the numerical issue, we need to separate the term: (2*pi)^D from
% the calculation precedure. We plug it back in only for the log-likelihood
pterm = - M* (N/2) * log(2*pi);

zz1 = lrpdf(data, mu1_old, sigma1_old, k)* prior_old(1);
zz2 = lrpdf(data, mu2_old, sigma2_old, k)* prior_old(2);
ll = sum(log(zz1+zz2)) + pterm;

ll_monitor = ll;

maxIter = 2;

%% start EM LOOP
for em = 1:20
    disp(['current em iter: ', num2str(em)])
    
    %% E step
    % U matrix: N-by-C , posterior matrix 

    U(:, 1) = prior_old(1)* lrpdf(data, mu1_old, sigma1_old, k);
    U(:, 2) = prior_old(2)* lrpdf(data, mu2_old, sigma2_old, k);
    Umatrix = U./repmat(sum(U, 2), 1, 2);


    %% M step

    Nk = sum(Umatrix);
    mu1_new = Umatrix(:, 1)'*data /Nk(1);
    sigma1_new = zeros(N);
    for ii = 1:M
        zi = data(ii, :)- mu1_new;
        sigma1_new = sigma1_new + Umatrix(ii, 1)* (zi'*zi);
    end
    sigma1_new = sigma1_new /Nk(1);

    mu2_new = Umatrix(:, 2)'*data /Nk(2);
    sigma2_new = zeros(N);
    for ii = 1:M
        zi = data(ii, :)- mu2_new;
        sigma2_new = sigma2_new + Umatrix(ii, 2)* (zi'*zi);
    end
    sigma2_new = sigma2_new /Nk(2);
    prior_new = Nk/M;

    %% check likelihood
%     ll = 0;
      zz1 = lrpdf(data, mu1_new, sigma1_new, k)* prior_new(1);
      zz2 = lrpdf(data, mu2_new, sigma2_new, k)* prior_new(2);
    ll = sum(log(zz1+zz2)) + pterm;
    
    ll_monitor = [ll_monitor, ll]; 
    
    %% update parameter
    mu1_old = mu1_new; sigma1_old = sigma1_new;
    mu2_old = mu2_new; sigma2_old = sigma2_new;
    prior_old = prior_new;
    
    
end       

%%
plot(ll_monitor, '-o')
title('log-likelihood over each EM iteration')
xlabel('number of iteration')
ylabel('data log-likelihood')
%% plot the leaned mean image
figure;
subplot(2, 2, 1)
imagesc(reshape(mean(num2), 28, 28)); colormap(gray)
title('mean from the training data, #2')

subplot(2, 2, 2)
imagesc(reshape(mean(num6), 28, 28)); colormap(gray)
title('mean from the training data, #6')

subplot(2, 2, 3)
imagesc(reshape(mu1_new, 28, 28)); colormap(gray)
title('mean from the training data, #2')

subplot(2, 2, 4)
imagesc(reshape(mu2_new, 28, 28)); colormap(gray)
title('mean from the training data, #2')
suptitle('training image mean vs learnt mixture mean')


%% classification with posterior probability
% if p(z1|x) >= p(z2|x), claim x is class 1, 
% if p(z1|x) <  p(z2|x), claim x is class 2,

predicted = Umatrix(:, 1) >=Umatrix(:, 2);
nums = predicted *2 + (1-predicted)*6;
nums = nums';

accuracy = sum(nums == trueLabel)/M;
disp(['the overall accuracy for GMM: ', num2str(accuracy)]);

%% now compare with knn
[km_labels, km_centroids] = kmeans(data, 2);
km_labels = km_labels -1;
nums_km = km_labels *2 + (1-km_labels)*6;

accuracy_km = sum(nums_km' == trueLabel)/M;
disp(['the overall accuracy for Kmeans: ',num2str(accuracy_km)])

%% 
function p = lrpdf(x, mu, S, k)
% S:= covariance matrix, 
% k:= truncation parameter
% in this function, to avoid numerical issue, we drop the term (2pi)^D

% ndim = length(mu);
[u,d] = eig(S);
d = diag(d);
[~, idx] = sort(d, 'descend');

% z1 = 1/sqrt((2*pi)^ndim * prod(d(idx(1:k))));
z1 = 1/sqrt(prod(d(idx(1:k))));

z2 = 0;
n = size(x, 1);
for ii = 1:k    % truncate by k
    z2 = z2 + ((x-repmat(mu, n, 1))*u(:, idx(ii))).^2/d(idx(ii));
end
z3 = exp(-1/2*z2);
p = z1*z3;
end
    
    
    
