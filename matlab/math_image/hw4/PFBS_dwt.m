%% this script is to complement PFBS algorithm with dwt
% coder: Jie An
% version: 20170504
% bug_submission: pkuanjie@gmail.com

clear
clc
addpath(genpath('./2DTWFT'));

%% set up
KERNEL_SIZE = 15;
SHIFT_LEN = (KERNEL_SIZE - 1) / 2;
SIGMA = 1.5;
SCALE = 200000;
LAMBDA = 0.000001;
KAPPA = 1;
L = 0.2 + KAPPA;
MAX_ITERATION = 800;
PATH_OF_IMAGE = 'aircraft.jpg';
FRAME = 1;
LEVEL = 2;
[D,R]=GenerateFrameletFilter(FRAME);
W  = @(x) FraDecMultiLevel2D(x,D,LEVEL); % Frame decomposition
WT = @(x) FraRecMultiLevel2D(x,R,LEVEL); % Frame reconstruction


%% image pre processing
disp('start image preprocessing');
img = imread(PATH_OF_IMAGE);
if size(size(img), 2) == 3
    img = im2double(rgb2gray(img));
else
    img = im2double(img);
end;
img = imresize(img, [200, 200]);
% figure,
% imshow(img);
kernel = fspecial('gaussian', [KERNEL_SIZE, KERNEL_SIZE], SIGMA);
img_blur = imfilter(img, kernel, 'circular');
% figure,
% imshow(img_blur);
img_pro = img_blur + max(max(img)) / SCALE * randn(size(img, 1));
% figure,
% imshow(img_pro);


%% PFBS
alpha = W(img_pro);
f = img_pro;
figure,
for i = 1 : MAX_ITERATION
    tmp_g = gradient_F2(alpha, f, W, WT, kernel);
    for x = 1 : 3
        for y = 1 : 3
            g{1}{x, y} = alpha{1}{x, y} - tmp_g{1}{x, y} / L;
            g{2}{x, y} = alpha{2}{x, y} - tmp_g{2}{x, y} / L;
        end;
    end;
    tmp = g;
    tmp3 = zeros(size(img));
    for x = 1 : 3
        for y = 1 : 3
            tmp3 = tmp3 + tmp{1}{x, y} .^ 2 + tmp{2}{x, y} .^ 2;
        end;
    end;
    tmp3 = sqrt(tmp3);
    TAU = LAMBDA / L;
    for x = 1 : 3
        for y = 1 : 3
            tmp{1}{x, y}(find(tmp3 > TAU)) = tmp{1}{x, y}(find(tmp3 > TAU)) ./ tmp3(find(tmp3 > TAU)) .* (tmp3(find(tmp3 > TAU)) - TAU);
            tmp{1}{x, y}(find(tmp3 <= TAU)) = 0 * tmp{1}{x, y}(find(tmp3 <= TAU));
            tmp{2}{x, y}(find(tmp3 > TAU)) = tmp{2}{x, y}(find(tmp3 > TAU)) ./ tmp3(find(tmp3 > TAU)) .* (tmp3(find(tmp3 > TAU)) - TAU);
            tmp{2}{x, y}(find(tmp3 <= TAU)) = 0 * tmp{2}{x, y}(find(tmp3 <= TAU));
        end;
    end;
    
    alpha = tmp;
    u = WT(alpha);
    imshow(u);
    disp(['iteration', num2str(i), 'finished!']);
end;


%% result show
figure,
subplot(131), imshow(img), title('original image');
subplot(132), imshow(f), title('filtered image');
subplot(133), imshow(u), title('reconstructed image');

figure,
subplot(131), imagesc(img), axis image, title('original image'), colorbar;
subplot(132), imagesc(f), title('filtered image'), axis image, colorbar;
subplot(133), imagesc(u), title('reconstructed image'), axis image, colorbar;