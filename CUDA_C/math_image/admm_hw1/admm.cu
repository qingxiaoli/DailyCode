#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <stdio.h>
#include "iostream"
#include "cuda_runtime.h"
#include "admm_tools.cuh"
using namespace std;
using namespace cv;

int main(){
    const Size KERNEL_SIZE(15, 15);
    const double SIGMA = 1.5;
    const int SCALE = 100;
    const double LAMBDA = sqrt(2) - 1;

    // image resize, Gaussian blur and noise adding of original image

    Mat img = imread("lena.bmp");
    Mat img_gray_tmp;
    Mat_<double> img_gray;
    cvtColor(img, img_gray_tmp, CV_RGB2GRAY);
    img_gray_tmp.convertTo(img_gray, CV_64FC1, 1.0/255.0);
    img.release();
    img_gray_tmp.release();
    resize(img_gray, img_gray, Size(), 1.0 / 4, 1.0 / 4, INTER_LINEAR);
    //namedWindow("img1", WINDOW_NORMAL);
    //imshow("img1", img_gray);
    //waitKey(3000);
    GaussianBlur(img_gray, img_gray, KERNEL_SIZE, SIGMA, BORDER_WRAP);
    //namedWindow("img2", WINDOW_NORMAL);
    //imshow("img2", img_gray);
    //waitKey(3000);
    double* value_max_img = new double[1];
    cuda::minMax(img_gray, NULL, value_max_img);
    Mat_<double> img_noise(img_gray.rows, img_gray.cols, CV_64FC1);
    randn(img_noise, 0, value_max_img[0]);
    cuda::multiply(img_noise, value_max_img[0] / SCALE, img_noise);
    cuda::add(img_gray, img_noise, img_gray);
    //namedWindow("img3", WINDOW_NORMAL);
    //imshow("img3", img_gray);
    //waitKey(3000);
    img_noise.release();
    free(value_max_img);

    // generate matrix A and W

    Mat_<double> A = Mat::zeros(img_gray.rows * img_gray.cols, img_gray.rows * img_gray.cols, CV_64FC1);
    if (img_gray.isContinuous() == 0){
        throw "memory of gray image is not continuous, cannot use cuda!";
    }
    Mat_<double> W = Mat::zeros(img_gray.rows * img_gray.cols, img_gray.rows * img_gray.cols, CV_64FC1);
    if (img_gray.isContinuous() == 0){
        throw "memory of gray image is not continuous, cannot use cuda!";
    }
    double* A_host = A.ptr<double>(0);
    double* A_device;
    cudaMalloc((void**)&A_device, sizeof(double) * A.cols * A.rows);
    cudaMemcpy(A_device, A_host, sizeof(double) * A.cols * A.rows, cudaMemcpyHostToDevice);
    double* W_host = W.ptr<double>(0);
    double* W_device;
    cudaMalloc((void**)&W_device, sizeof(double) * W.cols * W.rows);
    cudaMemcpy(W_device, W_host, sizeof(double) * W.cols * W.rows, cudaMemcpyHostToDevice);
    Mat_<double> Gauss = Mat::zeros(15, 15, CV_64FC1);
    Gauss.at<double>(7, 7) = 1;
    GaussianBlur(Gauss, Gauss, KERNEL_SIZE, SIGMA, BORDER_WRAP);
    double* Gauss_device;
    cudaMalloc((void**)&Gauss_device, sizeof(double) * Gauss.cols * Gauss.rows);
    cudaMemcpy(Gauss_device, Gauss.ptr<double>(0), sizeof(double) * Gauss.cols * Gauss.rows, cudaMemcpyHostToDevice);
    dim3 thread_perblock(Gauss.rows, Gauss.cols);
    compute_A<<<A.rows, thread_perblock>>>(A_device, Gauss_device, A.rows, A.cols, img_gray.rows, img_gray.cols, Gauss.rows, Gauss.cols);
    compute_W<<<W.rows, 1>>>(W_device, W.rows, W.cols, img_gray.rows, img_gray.cols, LAMBDA);
    cudaMemcpy(A_host, A_device, sizeof(double) * A.cols * A.rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(W_host, W_device, sizeof(double) * W.cols * W.rows, cudaMemcpyDeviceToHost);
    cudaFree(Gauss_device);
    cudaFree(A_device);
    cudaFree(W_device);
    Gauss.release();
}
