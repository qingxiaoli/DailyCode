#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudev.hpp>
#include <stdio.h>
#include "iostream"
#include "cuda_runtime.h"
#include "admm_tools.cuh"
using namespace std;
using namespace cv;

int main(){
    const Size KERNEL_SIZE(15, 15);
    const float SIGMA = 1.5;
    const int SCALE = 100;
    const float LAMBDA = sqrt(2) - 1;

    // image resize, Gaussian blur and noise adding of original image

    cout << "pre processing of image" << endl;
    Mat img = imread("lena.bmp");
    Mat img_gray_tmp;
    Mat_<float> img_gray;
    cvtColor(img, img_gray_tmp, CV_RGB2GRAY);
    img_gray_tmp.convertTo(img_gray, CV_32FC1, 1.0/255.0);
    img.release();
    img_gray_tmp.release();
    resize(img_gray, img_gray, Size(), 1.0 / 4, 1.0 / 4, INTER_LINEAR);
    Mat_<float> img_ori;
    img_gray.copyTo(img_ori);
    //namedWindow("img1", WINDOW_NORMAL);
    //imshow("img1", img_gray);
    //waitKey(0);
    GaussianBlur(img_gray, img_gray, KERNEL_SIZE, SIGMA, SIGMA, BORDER_WRAP);
    //namedWindow("img2", WINDOW_NORMAL);
    //imshow("img2", img_gray);
    //waitKey(3000);
    double* value_max_img = new double[1];
    cuda::minMax(img_gray, NULL, value_max_img);
    Mat_<float> img_noise(img_gray.rows, img_gray.cols, CV_32FC1);
    randn(img_noise, 0, value_max_img[0]);
    cuda::multiply(img_noise, value_max_img[0] / SCALE, img_noise);
    cuda::add(img_gray, img_noise, img_gray);
    //namedWindow("img3", WINDOW_NORMAL);
    //imshow("img3", img_gray);
    //waitKey(3000);
    img_noise.release();
    free(value_max_img);

    // Split Bregman Algorithm iteration

    cout << "prepare iteration" << endl;
    const float mu = 0.5;
    const float tol = 1e-6;
    const float delta = 0.5;
    const int MAX_ITERATION = 100;
    const float L = 1;

    Mat_<double> A = Mat_<double>::zeros(KERNEL_SIZE);
    Mat_<double> Lap = Mat_<double>::zeros(KERNEL_SIZE);
    A.at<double>((KERNEL_SIZE.height - 1) / 2, (KERNEL_SIZE.width - 1) / 2) = 1;
    cout << A << endl;
    GaussianBlur(A, A, KERNEL_SIZE, SIGMA, SIGMA, BORDER_WRAP);
    cout << A << endl;
    Mat_<double> z0 = Mat_<double>::zeros(img_gray.rows, img_gray.cols);
    cudev::GpuMat_<double> d(z0);
    cudev::GpuMat_<double> b(z0);
    cudev::GpuMat_<double> u(img_gray);
    Mat_<double> f;
    //GaussianBlur(f, f, KERNEL_SIZE, SIGMA, 0, BORDER_WRAP);

    // result show

    //Mat_<float> result;
    //Mat_<float> result_tmp;
    //u.download(result_tmp);
    //result = result_tmp.reshape(0, img_gray.rows);
    //result_tmp.release();
    //transpose(result, result);
    //namedWindow("result show", WINDOW_NORMAL);
    //imshow("result show", result);
    //waitKey(0);
    //float error = norm(result, img_ori, NORM_L2);
    //cout << "error of processing: " << error << endl;
//
    //// release memory
//
    //A_device.release();
    //W_device.release();
    //d.release();
    //b.release();
    //u.release();
    //tmp1.release();
    //tmp2.release();
    //tmp3.release();
    //tmp4.release();
    //img_gray.release();
    //img_ori.release();
    //result.release();
}
