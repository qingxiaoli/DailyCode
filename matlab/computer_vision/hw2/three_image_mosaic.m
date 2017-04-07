%% this script is to complement 3 image mosaic with SIFT and RANSAC
% coder: Jie An
% version: 20170406
% bug_submission: pkuanjie@gmail.com

%% set up
PATH1 = '1.JPG';
PATH2 = '2.JPG';
PATH3 = '3.JPG';
PATH4 = '4.JPG';

img1 = rgb2gray(imread(PATH1));
img2 = rgb2gray(imread(PATH2));
img3 = rgb2gray(imread(PATH3));
img4 = rgb2gray(imread(PATH4));

img1 = imresize(img1, [400, 600]);
img2 = imresize(img2, [400, 600]);
img3 = imresize(img3, [400, 600]);
img4 = imresize(img4, [400, 600]);

mean1 = mean(mean(img1));
mean2 = mean(mean(img2));
mean3 = mean(mean(img3));
mean4 = mean(mean(img4));

m = mean([mean1, mean2, mean3, mean4]);

img1 = single(img1 * (m / mean1));
img2 = single(img2 * (m / mean2));
img3 = single(img3 * (m / mean3));
img4 = single(img4 * (m / mean4));

mosaic1 = two_image_mosaic(img1, img2);
mosaic1 = single(mosaic1);
mosaic2 = two_image_mosaic(mosaic1, img3);
mosaic2 = single(mosaic2);
mosaic3 = two_image_mosaic(mosaic2, img4);
