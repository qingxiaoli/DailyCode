%% two image mosaic test
% coder: Jie An
% version: 20170407
% bug_submission: pkuanjie@gmail.com

%% import images
PATH1 = 'uttower_left.jpg';
PATH2 = 'uttower_right.jpg';

img1 = rgb2gray(imread(PATH1));
img2 = rgb2gray(imread(PATH2));

mean1 = mean(mean(img1));
mean2 = mean(mean(img2));

m = mean([mean1, mean2]);

img1 = single(img1 * (m / mean1));
img2 = single(img2 * (m / mean2));

%% mosaic
mosaic1 = two_image_mosaic(img1, img2);