%% two image mosaic test
% coder: Jie An
% version: 20170407
% bug_submission: pkuanjie@gmail.com

%% import images
PATH1 = 'uttower_left.jpg';
PATH2 = 'uttower_right.jpg';
img1 = imread(PATH1);
img2 = imread(PATH2);

mean1 = mean(mean(double(rgb2gray(img1))));
mean2 = mean(mean(double(rgb2gray(img2))));

m = mean([mean1, mean2]);

for i = 1 : 3
    img1(:, :, i) = single(img1(:, :, i) * (m / mean1));
    img2(:, :, i) = single(img2(:, :, i) * (m / mean2));
end;

%% mosaic
mosaic = two_image_mosaic(img1, img2);
figure, 
imshow(mosaic),
title('mosaic result');

%% old and new mosaic test failed
% %% import images
% PATH1 = 'spring_left.JPG';
% PATH2 = 'winter_middle.jpg';
% img1 = imread(PATH1);
% img2 = imread(PATH2);
% img1 = imresize(img1, [400, 600]);
% img2 = imresize(img2, [400, 600]);
% 
% mean1 = mean(mean(double(rgb2gray(img1))));
% mean2 = mean(mean(double(rgb2gray(img2))));
% 
% m = mean([mean1, mean2]);
% 
% for i = 1 : 3
%     img1(:, :, i) = single(img1(:, :, i) * (m / mean1));
%     img2(:, :, i) = single(img2(:, :, i) * (m / mean2));
% end;
% 
% %% mosaic
% mosaic = two_image_mosaic(img1, img2);
% figure, 
% imshow(mosaic),
% title('mosaic result');
