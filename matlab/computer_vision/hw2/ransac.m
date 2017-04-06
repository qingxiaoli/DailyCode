%% this function is to find a good model parameters with RANSAC algo
function [best_model, best_consensus_set_A, best_consensus_set_B, best_error] = ransac(A, B, N, K, T, D)
% parameters:
%     A: a matrix with input points coordinates to find model parameters;
%     B: a matrix with output points coordinates to find model parameters;
%     model: a function of model;
%     N: a int number, minimum data number to fit a model;
%     K: maximum itetation number;
%     T: threshold to test whether data satisify the model;
%     D: threshold to test whether model can be support confidently;
%     best_model: best model parameters;
%     best_consensus_set: inner point of best model;
%     best_error: error of best model;
% coder: Jie An
% version: 20170405
% bug_submission: pkuanjie@gmail.com

%% set up
best_model = ones(3, 3);
best_error = 1000000;
best_consensus_set_A = [];
best_consensus_set_B = [];

%% ransac iteration
for i = 1 : K
    choose = round((size(A, 1) - 1) * rand(N, 1)) + 1;
    A_piece = A(choose, :);
    B_piece = B(choose, :);
    A_piece_extend = [A_piece, ones(size(A_piece, 1), 1)];
    B_piece_extend = [B_piece, ones(size(B_piece, 1), 1)];
    A_extend = [A, ones(size(A, 1), 1)];
    B_extend = [B, ones(size(B, 1), 1)];
    maybe_model = pinv(A_piece_extend) * B_piece_extend;
    error = sqrt(diag((B_extend - A_extend * maybe_model) * (B_extend - A_extend * maybe_model)'));
    consensus_set_A = A_extend(error < T, :);
    consensus_set_B = B_extend(error < T, :);
    if size(consensus_set_A, 1) > D
        better_model = pinv(A_piece_extend) * B_piece_extend;
        set_error = diag((consensus_set_B - consensus_set_A * better_model) * (consensus_set_B - consensus_set_A * better_model)');
        this_error = mean(set_error);
        if this_error < best_error
            best_model = better_model;
            best_consensus_set_A = consensus_set_A;
            best_consensus_set_B = consensus_set_B;
            best_error =  this_error;
        end;
    end;
end;
