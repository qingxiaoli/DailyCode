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
    Mat_<double> img_ori;
    img_gray.copyTo(img_ori);
    //namedWindow("img1", WINDOW_NORMAL);
    //imshow("img1", img_gray);
    //waitKey(0);
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
    Mat_<double> W = Mat::zeros(img_gray.rows * img_gray.cols, img_gray.rows * img_gray.cols, CV_64FC1);
    cudev::GpuMat_<double> A_device(A);
    cudev::GpuMat_<double> W_device(W);
    if (A_device.isContinuous() == 0){
        throw "memory of gray image is not continuous, cannot use cuda!";
    }
    if (W_device.isContinuous() == 0){
        throw "memory of gray image is not continuous, cannot use cuda!";
    }
    Mat_<double> Gauss = Mat::zeros(15, 15, CV_64FC1);
    Gauss.at<double>(7, 7) = 1;
    GaussianBlur(Gauss, Gauss, KERNEL_SIZE, SIGMA, BORDER_WRAP);
    cudev::GpuMat_<double> Gauss_device(Gauss);
    dim3 thread_perblock(Gauss.rows, Gauss.cols);
    compute_A<<<A.rows, thread_perblock>>>(A_device.ptr<double>(0), Gauss_device.ptr<double>(0), A.rows, A.cols, img_gray.rows, img_gray.cols, Gauss.rows, Gauss.cols);
    compute_W<<<W.rows, 1>>>(W_device.ptr<double>(0), W.rows, W.cols, img_gray.rows, img_gray.cols, LAMBDA);
    Gauss.release();
    Gauss_device.release();

    // Split Bregman Algorithm iteration

    const double mu = 0.5;
    const double tol = 1e-6;
    const double delta = 0.5;
    const int MAX_ITERATION = 1000;
    Mat_<double> img_gray_t;
    transpose(img_gray, img_gray_t);
    Mat_<double> img_gray_t2 = img_gray_t.reshape(0, img_gray.rows * img_gray.cols);
    cudev::GpuMat_<double> f(img_gray_t2);
    cudev::GpuMat_<double> f_modified(img_gray_t2);
    img_gray_t.release();
    img_gray_t2.release();
    Mat_<double> source = Mat::zeros(f.rows, f.cols, CV_64FC1);
    cudev::GpuMat_<double> d(source);
    cudev::GpuMat_<double> b(source);
    cudev::GpuMat_<double> u(source);
    cudev::GpuMat_<double> tmp1(source);
    cudev::GpuMat_<double> tmp2(source);
    cudev::GpuMat_<double> tmp3(source);
    source.release();
    cuda::gemm(A_device, f, 1.0, 0, 1.0, f, GEMM_1_T);
    cuda::gemm(A_device, A_device, 1.0, 0, 1.0, A_device, GEMM_1_T);
    cuda::gemm(W_device, W_device, mu, A_device, 1.0, A_device, GEMM_1_T);
    A_device.download(A);
    invert(A, A, DECOMP_CHOLESKY);
    A_device.upload(A);
    cuda::gemm(A_device, f, 1.0, 0, 1.0, f, 0);
    cuda::gemm(A_device, W_device, 1.0, 0, mu, A_device, GEMM_2_T);


    for (int i = 0; i < MAX_ITERATION; i++)
    {
        cuda::addWeighted(d, 1.0, b, -1.0, 1.0, tmp1);
        cuda::gemm(A_device, tmp1, 1.0, f, 1.0, u);
        cuda::gemm(W_device, u, 1.0, b, 1.0, tmp2);
    }








    // result show

    Mat_<double> result;
    Mat_<double> result_tmp;
    d.download(result_tmp);
    result = result_tmp.reshape(0, img_gray.rows);
    result_tmp.release();
    transpose(result, result);
    namedWindow("result show", WINDOW_NORMAL);
    imshow("result show", result);
    waitKey(0);


    // release memory

    A_device.release();
    W_device.release();
}
