%% this script is to perform image denoising with Perona Malik method
% coder: Jie An
% version: 20170321
% bug_submission: pkuanjie@gmail.com


%% set up
IMG_PATH = 'lena.bmp';
MAX_ITERATION = [100, 100, 100, 1000];
NOISE_SCALE = 2;
DISCRETE_TIME = 0.05;
K = 0.05;



%% image preprocessing
img = im2double(rgb2gray(imread(IMG_PATH)));
% figure,
% imshow(img), axis image;
img_noise = img + max(max(img)) / NOISE_SCALE * randn(size(img));
% figure,
% imshow(img_noise), axis image;


%% perona malik equation
img_pro = img_noise;
result = cell(1, size(MAX_ITERATION, 2));
% figure, 
for cycle = 1 : size(MAX_ITERATION, 2)
    for i = 1 : MAX_ITERATION(cycle)
        [dx, dy] = gradient(img_pro);
        tmp = dx .^ 2 + dy .^ 2;
        tmp = 1 ./ (1 + tmp / (K ^ 2));
        dx = dx .* tmp;
        dy = dy .* tmp;
        [ddx, ~] = gradient(dx);
        [~, ddy] = gradient(dy);
        tmp = ddx + ddy;
        img_pro = img_pro + DISCRETE_TIME * tmp;
%         imshow(img_pro), axis image;
        disp(['num = ', num2str(i), ' image processing finished']);
    end;
    result{cycle} = img_pro;
end;


%% result show
figure,
subplot(131), imshow(img), axis image, title('original image');
subplot(132), imshow(img_noise), axis image, title('noised image');
subplot(133), imshow(result{1}), axis image, title('denoised image');

figure,
subplot(221), imshow(result{1}), axis image, title(['terminate in ', num2str(MAX_ITERATION(1))]);
subplot(222), imshow(result{2}), axis image, title(['terminate in ', num2str(MAX_ITERATION(2))]);
subplot(223), imshow(result{3}), axis image, title(['terminate in ', num2str(MAX_ITERATION(3))]);
subplot(224), imshow(result{4}), axis image, title(['terminate in ', num2str(MAX_ITERATION(4))]);