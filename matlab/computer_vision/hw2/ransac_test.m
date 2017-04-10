%% this script is to test ransac algo
% coder: Jie An
% version: 20170405
% bug_submission: pkuanjie@gmail.com

%% naive linear test
% N = 2;
% K = 100;
% T = 1;
% D = 95;
% x = 10 * rand(100, 1);
% y = x * 2 + 3 + 0.2 * rand(100, 1) - 0.1;
% noise = zeros(50, 2);
% noise(:, 1) = 10 * rand(50, 1);
% noise(:, 2) = 25 * rand(50, 1);
% data = [[x, y]; noise];
% % figure,
% % plot(data(:, 1), data(:, 2), '.');
% 
% [best_model, best_consensus_set, best_error] = ransac_linear(data, @linear_model, @linear_model_error, N, K, T, D);

%% homography test
N = 4;
K = 200;
T = 1;
D = 90;
X = [0.6, 0.7, 0.8; 0.5, 0.6, 0.7; 0.4, 0.5, 1];
A = 10 * rand(100, 2);
tmp = [A, ones(size(A, 1), 1)] * X;
B = tmp(:, 1: 2) ./ repmat(tmp(:, 3), 1, 2);
figure,
plot(A(:, 1), A(:, 2), '.');
title('original A scatter');
figure,
plot(B(:, 1), B(:, 2), '.');
title('original B scatter');
[best_model, best_consensus_set_A, best_consensus_set_B, best_error] = ransac(A, B, N, K, T, D);
tmp = [A, ones(size(A, 1), 1)] * best_model;
B_pre = [A, ones(size(A, 1), 1)] * best_model ./ repmat(tmp(:, 3), [1, 3]);
figure, 
plot(B_pre(:, 1), B_pre(:, 2), '.');
title('predict B scatter');


