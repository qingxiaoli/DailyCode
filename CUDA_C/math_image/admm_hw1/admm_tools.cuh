#ifndef ADMM_TOOLS_CUH
#define ADMM_TOOLS_CUH

__global__ void compute_A(double*, double*, int, int, int, int, int, int);
__global__ void compute_W(double*, int, int, int, int, double);

#endif
