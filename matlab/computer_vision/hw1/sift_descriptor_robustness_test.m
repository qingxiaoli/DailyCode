%% this script is to test sift descriptor's robustness
% coder: Jie An
% version: 20170320
% bug_submission: pkuanjie@gmail.com

%% set up
IMG_PATH = 'building.jpg';
PEAK_THRESH = 1;
EDGE_THRESH = 5;
ROTATE_ANGLE = 15;
SCALE_FACTOR = 1.2;
BRIGHTNESS_MINUS_MAX = -100;
BRIGHTNESS_PLUS_MAX = 100;
BRIGHTNESS_CHANGE_LEVEL = 20;
CONTRAST_MIN = 0.5;
CONTRAST_MAX = 2.0;
CONTRAST_CHANGE_LEVEL = 0.25;
NOISE_MIN = 0;
NOISE_MAX = 30;
NOISE_CHANGE_LEVEL = 5;
GAUSS_MIN = 1;
GAUSS_MAX = 10;
GAUSS_CHANGE_LEVEL = 1;



%% img import
img = single(rgb2gray(imread(IMG_PATH)));
figure,
imagesc(img), axis image;


%% using sift to detect feature points
[frames, descriptors] = vl_sift(img, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH);
vl_plotframe(frames);
feature_ori = frames;
descriptors_ori = descriptors;


%% test robustness of sift descriptor on brightness
feature_matches_brightness = zeros(size(BRIGHTNESS_MINUS_MAX : BRIGHTNESS_CHANGE_LEVEL : BRIGHTNESS_PLUS_MAX, 2), 1);
for i = BRIGHTNESS_MINUS_MAX : BRIGHTNESS_CHANGE_LEVEL : BRIGHTNESS_PLUS_MAX
    img_brightness = img + i;
    img_brightness(find(img_brightness > 255)) = 255;
%     figure,
%     imagesc(img_brightness);
    [~, descriptor_brightness] = vl_sift(img_brightness, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH, 'Frames', frames);
    M = vl_ubcmatch(descriptors_ori, descriptor_brightness);
    feature_matches_brightness((i - BRIGHTNESS_MINUS_MAX + BRIGHTNESS_CHANGE_LEVEL) / BRIGHTNESS_CHANGE_LEVEL) = size(M, 2) / size(descriptors_ori, 2);
end;
figure,
plot(BRIGHTNESS_MINUS_MAX : BRIGHTNESS_CHANGE_LEVEL : BRIGHTNESS_PLUS_MAX, feature_matches_brightness, 'ro'), title('repeatability of brightness');
xlabel('brightness'),
ylabel('descriptor matches rate'),
set(gca, 'XTick', BRIGHTNESS_MINUS_MAX : BRIGHTNESS_CHANGE_LEVEL : BRIGHTNESS_PLUS_MAX);


%% test robustness of sift descriptor on contrast
feature_matches_contrast = zeros(size(CONTRAST_MIN : CONTRAST_CHANGE_LEVEL : CONTRAST_MAX, 2), 1);
for i = CONTRAST_MIN : CONTRAST_CHANGE_LEVEL : CONTRAST_MAX
    img_contrast = img * i;
    img_contrast(find(img_contrast > 255)) = 255;
%     figure,
%     imagesc(img_contrast);
    [~, descriptor_contrast] = vl_sift(img_contrast, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH, 'Frames', frames);
    M = vl_ubcmatch(descriptors_ori, descriptor_contrast);
    feature_matches_contrast((i - CONTRAST_MIN + CONTRAST_CHANGE_LEVEL) / CONTRAST_CHANGE_LEVEL) = size(M, 2) / size(descriptors_ori, 2);
end;
figure,
plot(CONTRAST_MIN : CONTRAST_CHANGE_LEVEL : CONTRAST_MAX, feature_matches_contrast, 'ro'), title('repeatability of contrast');
xlabel('contrast'),
ylabel('descriptor matches rate'),
set(gca, 'XTick', CONTRAST_MIN : CONTRAST_CHANGE_LEVEL : CONTRAST_MAX);


%% test robustness of sift descriptor on noise
feature_matches_noise = zeros(size(NOISE_MIN : NOISE_CHANGE_LEVEL : NOISE_MAX, 2), 1);
for i = NOISE_MIN : NOISE_CHANGE_LEVEL : NOISE_MAX
    img_noise = img + i * randn(size(img));
    img_noise(find(img_noise > 255)) = 255;
    img_noise(find(img_noise < 0)) = 0;
%     figure,
%     imagesc(img_noise);
    [~, descriptor_noise] = vl_sift(img_noise, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH, 'Frames', frames);
    M = vl_ubcmatch(descriptors_ori, descriptor_noise);
    feature_matches_noise((i - NOISE_MIN + NOISE_CHANGE_LEVEL) / NOISE_CHANGE_LEVEL) = size(M, 2) / size(descriptors_ori, 2);
end;
figure,
plot(NOISE_MIN : NOISE_CHANGE_LEVEL : NOISE_MAX, feature_matches_noise, 'ro'), title('repeatability of noise');
xlabel('noise'),
ylabel('descriptor matches rate'),
set(gca, 'XTick', NOISE_MIN : NOISE_CHANGE_LEVEL : NOISE_MAX);



%% test robustness of sift descriptor on blur
feature_matches_blur = zeros(size(GAUSS_MIN : GAUSS_CHANGE_LEVEL : GAUSS_MAX, 2), 1);
for i = GAUSS_MIN : GAUSS_CHANGE_LEVEL : GAUSS_MAX
    kernel = fspecial('gaussian', [10 * i + 1, 10 * i + 1], i);
    img_blur = imfilter(img, kernel, 'circular');
    img_blur(find(img_blur > 255)) = 255;
    img_blur(find(img_blur < 0)) = 0;
%     figure,
%     imagesc(img_blur);
    [~, descriptor_blur] = vl_sift(img_blur, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH, 'Frames', frames);
    M = vl_ubcmatch(descriptors_ori, descriptor_blur);
    feature_matches_blur((i - GAUSS_MIN + GAUSS_CHANGE_LEVEL) / GAUSS_CHANGE_LEVEL) = size(M, 2) / size(descriptors_ori, 2);
end;
figure,
plot(GAUSS_MIN : GAUSS_CHANGE_LEVEL : GAUSS_MAX, feature_matches_blur, 'ro'), title('repeatability of blur');
xlabel('blur'),
ylabel('descriptor matches rate'),
set(gca, 'XTick', GAUSS_MIN : GAUSS_CHANGE_LEVEL : GAUSS_MAX);

