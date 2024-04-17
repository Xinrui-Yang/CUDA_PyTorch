#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BLOCK_WIDTH 16

__global__ void MatrixMulKernel(float *A, float *B, float *C, int rows_A, int cols_A, int cols_B)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rows_A && col < cols_B)
	{
		float value = 0;
		for (int i = 0; i < cols_A; i++)
		{
			value += A[row * cols_A + i] * B[i * cols_B + col];
		}
		C[row * cols_B + col] = value;
	}
}

void MatrixMul_cuda(float *A, float *B, float *C, int rows_A, int cols_A, int cols_B)
{
	float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows_A*cols_A*sizeof(float));
    cudaMalloc((void**)&d_B, cols_A*cols_B*sizeof(float));
	cudaMalloc((void**)&d_C, rows_A*cols_B*sizeof(float));

	cudaMemcpy(d_A, A, rows_A*cols_A*sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(d_B, B, cols_A*cols_B*sizeof(float), cudaMemcpyHostToDevice);

	dim3 grid((cols_B + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (rows_A + BLOCK_WIDTH - 1) / BLOCK_WIDTH);
	dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
	MatrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, rows_A, cols_A, cols_B);

	cudaMemcpy(C, d_C, rows_A*cols_B*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}