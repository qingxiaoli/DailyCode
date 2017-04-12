%% this script is to complement 3 image mosaic with SIFT and RANSAC
% coder: Jie An
% version: 20170406
% bug_submission: pkuanjie@gmail.com

%% set up
PATH1 = '1.JPG';
PATH2 = '2.JPG';
PATH3 = '3.JPG';

img1_rgb = imread(PATH1);
img2_rgb = imread(PATH2);
img3_rgb = imread(PATH3);

img1_rgb = imresize(img1_rgb, [400, 600]);
img2_rgb = imresize(img2_rgb, [400, 600]);
img3_rgb = imresize(img3_rgb, [400, 600]);

img1 = rgb2gray(img1_rgb);
img2 = rgb2gray(img2_rgb);
img3 = rgb2gray(img3_rgb);

mean1 = mean(mean(img1));
mean2 = mean(mean(img2));
mean3 = mean(mean(img3));

m = mean([mean1, mean2, mean3]);

img1 = single(img1 * (m / mean1));
img2 = single(img2 * (m / mean2));
img3 = single(img3 * (m / mean3));

mosaic1 = two_image_mosaic(img1_rgb, img2_rgb);
mosaic2 = two_image_mosaic(mosaic1, img3_rgb);

%% set up
PATH1 = 'spring_left.JPG';
PATH2 = 'winter_middle.jpg';
PATH3 = 'spring_right.JPG';

img1_rgb = imread(PATH1);
img2_rgb = imread(PATH2);
img3_rgb = imread(PATH3);

img1_rgb = imresize(img1_rgb, [400, 600]);
img2_rgb = imresize(img2_rgb, [400, 600]);
img3_rgb = imresize(img3_rgb, [400, 600]);

img1 = rgb2gray(img1_rgb);
img2 = rgb2gray(img2_rgb);
img3 = rgb2gray(img3_rgb);

mean1 = mean(mean(img1));
mean2 = mean(mean(img2));
mean3 = mean(mean(img3));

m = mean([mean1, mean2, mean3]);

img1 = single(img1 * (m / mean1));
img2 = single(img2 * (m / mean2));
img3 = single(img3 * (m / mean3));

mosaic1 = two_image_mosaic(img1_rgb, img2_rgb);
mosaic2 = two_image_mosaic(mosaic1, img3_rgb);
