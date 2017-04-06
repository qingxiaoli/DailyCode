%% function of model
function para = linear_model(data_piece)
% coder: Jie An
% version: 20170405
% bug_submission: pkuanjie@gmail.com

para = zeros(2, 1);
para(1) = (data_piece(2, 2) - data_piece(1, 2)) / (data_piece(2, 1) - data_piece(1, 1));
para(2) = data_piece(1, 2) - data_piece(1, 1) * para(1);