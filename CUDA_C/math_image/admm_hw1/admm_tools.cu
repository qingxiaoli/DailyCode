#include "admm_tools.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void compute_A(double* A_device, double* Gauss_device, int rows, int cols, int ori_rows, int ori_cols, int Gauss_rows, int Gauss_cols)
{
    int j = blockIdx.x;
    int k1 = threadIdx.y;
    int k2 = threadIdx.x;
    int len_x = (Gauss_rows - 1) / 2;
    int len_y = (Gauss_cols - 1) / 2;
    if (j < cols)
    {
        int b = j / ori_rows;
        int a = j % ori_rows;
        int a1 = a - len_x;
        int b1 = b - len_y;
        int k_a = k1 + a1;
        int k_b = k2 + b1;
        if (k_a >= 0 && k_a < ori_rows && k_b >= 0 && k_b < ori_cols && k1 < Gauss_rows && k2 < Gauss_cols)
        {
          A_device[(k_b * ori_rows + k_a) * cols + j] = Gauss_device[k1 * Gauss_cols + k2];
        }// inner situation
        if (k_a < 0 && k_b >= 0 && k_b < ori_cols && k1 < Gauss_rows && k2 < Gauss_cols)
        {
          A_device[((ori_rows + k_a) + k_b * ori_rows) * cols + j] = Gauss_device[k1 * Gauss_cols + k2];
        }// top edge
        if (k_a >= ori_rows && k_b >= 0 && k_b < ori_cols && k1 < Gauss_rows && k2 < Gauss_cols)
        {
          A_device[((k_a - ori_rows) + k_b * ori_rows) * cols + j] = Gauss_device[k1 * Gauss_cols + k2];
        }// bottom edge
        if (k_a >= 0 && k_a < ori_rows && k_b < 0 && k1 < Gauss_rows && k2 < Gauss_cols)
        {
          A_device[(k_a + (k_b + ori_cols) * ori_rows) * cols + j] = Gauss_device[k1 * Gauss_cols + k2];
        }// laft edge
        if (k_a >= 0 && k_a < ori_rows && k_b >= ori_cols && k1 < Gauss_rows && k2 < Gauss_cols)
        {
          A_device[(k_a + (k_b - ori_cols) * ori_rows) * cols + j] = Gauss_device[k1 * Gauss_cols + k2];
        }// right edge
        if (k_a < 0 && k_b < 0 && k1 < Gauss_rows && k2 < Gauss_cols)
        {
          A_device[((k_a + ori_rows) + (k_b + ori_cols) * ori_rows) * cols + j] = Gauss_device[k1 * Gauss_cols + k2];
        }// left top corner
        if (k_a < 0 && k_b >= ori_cols && k1 < Gauss_rows && k2 < Gauss_cols)
        {
          A_device[((k_a + ori_rows) + (k_b - ori_cols) * ori_rows) * cols + j] = Gauss_device[k1 * Gauss_cols + k2];
        }// right top corner
        if (k_a >= ori_rows && k_b < 0 && k1 < Gauss_rows && k2 < Gauss_cols)
        {
          A_device[((k_a - ori_rows) + (k_b + ori_cols) * ori_rows) * cols + j] = Gauss_device[k1 * Gauss_cols + k2];
        }// left bottom corner
        if (k_a >= ori_rows && k_b >= ori_cols && k1 < Gauss_rows && k2 < Gauss_cols)
        {
          A_device[((k_a - ori_rows) + (k_b - ori_cols) * ori_rows) * cols + j] = Gauss_device[k1 * Gauss_cols + k2];
        }// right bottom corner
        __syncthreads();
    }
}


__global__ void compute_W(double* W_device, int rows, int cols, int ori_rows, int ori_cols, double LAMBDA)
{
    int j = blockIdx.x;
    int a = j / ori_rows;
    int b = j % ori_rows;
    if (a > 0 && a < ori_cols - 1 && b > 0 && b < ori_rows - 1)
    {
        W_device[(j - 1) * cols + j] = LAMBDA / 2.0;
        W_device[(j + 1) * cols + j] = LAMBDA / 2.0;
        W_device[(j - ori_rows) * cols + j] = LAMBDA / 2.0;
        W_device[(j + ori_rows) * cols + j] = LAMBDA / 2.0;
        W_device[(j - ori_rows - 1) * cols + j] = (1 - LAMBDA) * sqrt(2.0) / 4.0;
        W_device[(j - ori_rows + 1) * cols + j] = (1 - LAMBDA) * sqrt(2.0) / 4.0;
        W_device[(j + ori_rows - 1) * cols + j] = (1 - LAMBDA) * sqrt(2.0) / 4.0;
        W_device[(j + ori_rows + 1) * cols + j] = (1 - LAMBDA) * sqrt(2.0) / 4.0;
    }
}
