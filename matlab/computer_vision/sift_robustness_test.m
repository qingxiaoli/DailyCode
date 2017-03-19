%% this script is write to test the robustness of SIFT detector and descriptor
% coder: Jie An
% version: 20170319
% bug_submission: pkuanjie@gmail.com

%% set up
IMG_PATH = 'cover_1.jpg';

%% img import
img = single(rgb2gray(imread(IMG_PATH)));
figure,
imagesc(img), axis image;

%% using sift to detect feature points
[frames, descriptors] = vl_sift(img, 'PeakThresh', 1, 'EdgeThresh', 5);
vl_plotframe(frames);