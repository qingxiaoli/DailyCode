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
    while 1
        choose = unique(randperm(size(A, 1), 4));
        if (size(choose, 2) == 4)
            break;
        end;
    end;
    A_piece = A(choose, :);
    B_piece = B(choose, :);
    A_extend = [A, ones(size(A, 1), 1)];
    B_extend = [B, ones(size(B, 1), 1)];
    mat_data = zeros(8, 8);
    mat_right = zeros(8, 1);
    for j = 1: 2: 8
        mat_data(j, 1: 2) = A_piece((j + 1) / 2, :);
        mat_data(j, 3) = 1;
        mat_data(j, 7) = -B_piece((j + 1) / 2, 1) * A_piece((j + 1) / 2, 1);
        mat_data(j, 8) = -B_piece((j + 1) / 2, 1) * A_piece((j + 1) / 2, 2);
        mat_data(j + 1, 4: 5) = A_piece((j + 1) / 2, :);
        mat_data(j + 1, 6) = 1;
        mat_data(j + 1, 7) = -B_piece((j + 1) / 2, 2) * A_piece((j + 1) / 2, 1);
        mat_data(j + 1, 8) = -B_piece((j + 1) / 2, 2) * A_piece((j + 1) / 2, 2);
        mat_right(j) = B_piece((j + 1) / 2, 1);
        mat_right(j + 1) = B_piece((j + 1) / 2, 2);
    end;
    maybe_model = [mat_data \ mat_right; 1];
    maybe_model = reshape(maybe_model, [3, 3]);
    tmp = A_extend * maybe_model;
    error = sqrt(diag((B_extend - A_extend * maybe_model ./ repmat(tmp(:, 3), [1, 3])) * (B_extend - A_extend * maybe_model ./ repmat(tmp(:, 3), [1, 3]))'));
    consensus_set_A = A_extend(error < T, :);
    consensus_set_B = B_extend(error < T, :);
    if size(consensus_set_A, 1) > D
        mat_data = zeros(2 * size(consensus_set_A, 1), 8);
        mat_right = zeros(2 * size(consensus_set_A, 1), 1);
        for j = 1: 2: 2 * size(consensus_set_A, 1)
            mat_data(j, 1: 2) = consensus_set_A((j + 1) / 2, 1:2);
            mat_data(j, 3) = 1;
            mat_data(j, 7) = -consensus_set_B((j + 1) / 2, 1) * consensus_set_A((j + 1) / 2, 1);
            mat_data(j, 8) = -consensus_set_B((j + 1) / 2, 1) * consensus_set_A((j + 1) / 2, 2);
            mat_data(j + 1, 4: 5) = consensus_set_A((j + 1) / 2, 1:2);
            mat_data(j + 1, 6) = 1;
            mat_data(j + 1, 7) = -consensus_set_B((j + 1) / 2, 2) * consensus_set_A((j + 1) / 2, 1);
            mat_data(j + 1, 8) = -consensus_set_B((j + 1) / 2, 2) * consensus_set_A((j + 1) / 2, 2);
            mat_right(j) = consensus_set_B((j + 1) / 2, 1);
            mat_right(j + 1) = consensus_set_B((j + 1) / 2, 2);
        end;
        better_model = [mat_data \ mat_right; 1];
        better_model = reshape(better_model, [3, 3]);
        tmp = consensus_set_A * better_model;
        set_error = sqrt(diag((consensus_set_B - consensus_set_A * better_model ./ repmat(tmp(:, 3), [1, 3])) * (consensus_set_B - consensus_set_A * better_model ./ repmat(tmp(:, 3), [1, 3]))'));
        this_error = mean(set_error);
        if this_error < best_error
            best_model = better_model;
            best_consensus_set_A = consensus_set_A;
            best_consensus_set_B = consensus_set_B;
            best_error =  this_error;
        end;
    end;
end;
