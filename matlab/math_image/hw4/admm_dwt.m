%% this script is to use admm algorithm and discrete wavelet transform to implement image deblur
% coder: Jie An
% version: 20170426
% bug_submission: pkuanjie@gmail.com

clear
clc
addpath(genpath('./2DTWFT'));

%% set up
KERNEL_SIZE = 15;
SHIFT_LEN = (KERNEL_SIZE - 1) / 2;
SIGMA = 1.5;
SCALE = 200000;
MU = 1;
LAMBDA = 0.000001;
TAU = LAMBDA / MU;
TOL = 1e-15;
DELTA = 0.5;
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
kernel = fspecial('gaussian', [15, 15], SIGMA);
img_blur = imfilter(img, kernel, 'circular');
% figure,
% imshow(img_blur);
img_pro = img_blur + max(max(img)) / SCALE * randn(size(img, 1));
% figure,
% imshow(img_pro);

%% admm
A = kernel;
f = img_pro;
d = W(img_pro);
b = d;
Af = imfilter(f, A, 'circular');
tempA = zeros(size(img));
tempI = zeros(size(img));
tempA(1: KERNEL_SIZE, 1: KERNEL_SIZE) = A;
tempA = circshift(tempA, [-SHIFT_LEN, -SHIFT_LEN]);
tempI(1, 1) = MU;
ftA = fft2(tempA);
ftI = fft2(tempI);
tempFT = ftA .* ftA + ftI;% this mat is kernel to solve equation

error = zeros(MAX_ITERATION, 1);
figure,
for i = 1 : MAX_ITERATION

    % step 1
    for x = 1 : 3
        for y = 1 : 3
            tmp_d_b{1}{x, y} = d{1}{x, y} - b{1}{x, y};
            tmp_d_b{2}{x, y} = d{2}{x, y} - b{2}{x, y};
        end;
    end;
    u = real(ifft2(fft2(Af + MU * WT(tmp_d_b)) ./ (tempFT)));
    imshow(u);

    % step 2
    tmp_wu = W(u);
    for x = 1 : 3
        for y = 1 : 3
            tmp{1}{x, y} = tmp_wu{1}{x, y} + b{1}{x, y}; 
            tmp{2}{x, y} = tmp_wu{2}{x, y} + b{2}{x, y}; 
        end;
    end;

    tmp3 = zeros(size(img));
    for x = 1 : 3
        for y = 1 : 3
            tmp3 = tmp3 + tmp{1}{x, y} .^ 2 + tmp{2}{x, y} .^ 2;
        end;
    end;
    tmp3 = sqrt(tmp3);
    
    for x = 1 : 3
        for y = 1 : 3
            tmp{1}{x, y}(find(tmp3 > TAU)) = tmp{1}{x, y}(find(tmp3 > TAU)) ./ tmp3(find(tmp3 > TAU)) .* (tmp3(find(tmp3 > TAU)) - TAU);
            tmp{1}{x, y}(find(tmp3 <= TAU)) = 0 * tmp{1}{x, y}(find(tmp3 <= TAU));
            tmp{2}{x, y}(find(tmp3 > TAU)) = tmp{2}{x, y}(find(tmp3 > TAU)) ./ tmp3(find(tmp3 > TAU)) .* (tmp3(find(tmp3 > TAU)) - TAU);
            tmp{2}{x, y}(find(tmp3 <= TAU)) = 0 * tmp{2}{x, y}(find(tmp3 <= TAU));
        end;
    end;
    
    d = tmp;

    % step 3
    for x = 1 : 3
        for y = 1 : 3
            b{1}{x, y} = b{1}{x, y} + DELTA * (tmp_wu{1}{x, y} - d{1}{x, y});
            b{2}{x, y} = b{2}{x, y} + DELTA * (tmp_wu{2}{x, y} - d{2}{x, y});
        end;
    end;

    % error saving and break checking
    tmp4 = zeros(size(img));
    for x = 1 : 3
        for y = 1 : 3
            tmp4 = tmp4 + (tmp_wu{1}{x, y} - d{1}{x, y}) .^ 2 + (tmp_wu{2}{x, y} - d{2}{x, y}) .^ 2;
        end;
    end;
    tmp4 = sqrt(tmp4);
    check2 = norm(tmp4, 'fro') / norm(f, 'fro');
    error(i) = norm(u - img, 'fro') ^ 2;
    if check2 < TOL
        break;
    end;
    disp(['iteration num = ', num2str(i), ' finished. ', 'check error = ', num2str(check2)]);
end;

%% result show
figure,
subplot(221), imshow(img), title('original image');
subplot(222), imshow(f), title('filtered image');
subplot(223), imshow(u), title('reconstructed image');
subplot(224), plot(error), title('error');

figure,
subplot(131), imagesc(img), axis image, title('original image'), colorbar;
subplot(132), imagesc(f), title('filtered image'), axis image, colorbar;
subplot(133), imagesc(u), title('reconstructed image'), axis image, colorbar;
