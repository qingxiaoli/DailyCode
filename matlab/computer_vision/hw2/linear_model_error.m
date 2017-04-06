%% this function is to compute error of ransac model and data
function error = linear_model_error(maybe_model, data)
% coder: Jie An
% version: 20170405
% bug_submission: pkuanjie@gmail.com

%% compute
error = abs(data(:, 2) - (data(:, 1) * maybe_model(1) + maybe_model(2)));