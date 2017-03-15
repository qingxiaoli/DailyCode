#ifndef ADMM_TOOLS_CUH
#define ADMM_TOOLS_CUH

__global__ void compute_A(float*, float*, int, int, int, int, int, int);
__global__ void compute_W(float*, int, int, int, int, float);

#endif
