%% this script is to test robustness of Harris coner detection algorithm, test will be on rotation and scaling
% coder: Jie An
% version: 20170319
% bug_submission: pkuanjie@gmail.com
% license: no license, everyone is free to use

%% set up
IMG_PATH = 'cover_1.jpg';
MIN_QUALITY = 0.01;
ROTATE_ANGLE = 15;
SCALE_FACTOR = 1.2;


%% img import
img = im2double(rgb2gray(imread(IMG_PATH)));
% figure, 
% imagesc(img);


%% corner detection
corner_result = detectHarrisFeatures(img, 'MinQuality', MIN_QUALITY);
corner_metric = zeros(size(img));
for i = 1 : corner_result.Count
    corner_metric(round(corner_result.Location(i, 2)), round(corner_result.Location(i, 1))) = corner_result.Metric(i);
end;
% figure, 
% subplot(121), imagesc(img), axis image, hold on,
% plot(corner_result.Location(:, 1), corner_result.Location(:, 2), 'r.'),
% hold off;
% subplot(122), imagesc(corner_metric), axis image, colorbar;
corner_ori = corner_result;


%% rotate image and detect same corner point
corner_matches_rotation = zeros(size(ROTATE_ANGLE : ROTATE_ANGLE : 359, 2), 1);
for i = ROTATE_ANGLE : ROTATE_ANGLE : 359
    img_rotation = imrotate(img, i);
%     figure,
%     imagesc(img_rotation);
    rot_mat = [cos(i / 180 * pi), -sin(i / 180 * pi); sin(i / 180 * pi), cos(i / 180 * pi)];
    corners = detectHarrisFeatures(img_rotation, 'MinQuality', MIN_QUALITY);
    corners_rot = (corner_ori.Location -  size(img) / 2) * rot_mat;
    for j = 1 : corner_ori.Count
        if size(find(abs(corners.Location(:, 1) - size(img_rotation, 1) / 2 - corners_rot(j, 1)) < 2), 1) > 0 && size(find(abs(corners.Location(:, 2) - size(img_rotation, 2) / 2 - corners_rot(j, 2)) < 2), 1) > 0
            corner_matches_rotation(i / ROTATE_ANGLE) = corner_matches_rotation(i / ROTATE_ANGLE) + 1;
        end;
    end;
end;


%% scale image and detect same corner point
corner_matches_scale = zeros(9, 1);
for i = 0 : 8
    img_scale = imresize(img, SCALE_FACTOR ^ i);
%     figure,
%     imagesc(img_rotation);
    scale_mat = [SCALE_FACTOR ^ i, 0; 0, SCALE_FACTOR ^ i];
    corners = detectHarrisFeatures(img_scale, 'MinQuality', MIN_QUALITY);
    corners_sca = (corner_ori.Location -  size(img) / 2) * scale_mat;
    for j = 1 : corner_ori.Count
        if size(find(abs(corners.Location(:, 1) - size(img_scale, 1) / 2 - corners_sca(j, 1)) < 2), 1) > 0 && size(find(abs(corners.Location(:, 2) - size(img_scale, 2) / 2 - corners_sca(j, 2)) < 2), 1) > 0
            corner_matches_scale(i + 1) = corner_matches_scale(i +1) + 1;
        end;
    end;
end;
