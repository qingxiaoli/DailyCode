%% this function is to find a good model parameters with RANSAC algo
function [best_model, best_consensus_set, best_error] = ransac_linear(data, model, model_error, N, K, T, D)
% parameters:
%     data: a matrix with points coordinates to find model parameters;
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
best_model = zeros(2);
best_error = 1000000;
best_consensus_set = [];

%% ransac iteration
for i = 1 : K
    choose = round((size(data, 1) - 1) * rand(N, 1)) + 1;
    data_piece = data(choose, :);
    maybe_model = model(data_piece);
    error = linear_model_error(maybe_model, data);
    consensus_set = data(error < T, :);
    consensus_set = sortrows(consensus_set, 1);
    if size(consensus_set, 1) > D
        better_model = polyfit(consensus_set(:, 1), consensus_set(:, 2), 1);
        set_error = model_error(better_model, consensus_set);
        this_error = mean(set_error);
        if this_error < best_error
            best_model = better_model;
            best_consensus_set = consensus_set;
            best_error =  this_error;
        end;
    end;
end;
