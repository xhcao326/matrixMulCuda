
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

const int dimM = 512;
const int dimK = 512;
const int dimN = 512;

const int tileAOuter = 32;
const int tileInner = 32;
const int tileBOuter = 32;

const int workPerThread1 = 1;
const int workPerThread0 = 1;

const int workGroupSize1 = 16;
const int workGroupSize0 = 16;

cudaError_t matmulWithCuda(float* c, const float* a, const float* b, int M, int K, int N);

__global__ void matmulNaiveKernel(float* c, const float* a, const float* b, int M, int K, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float res = 0.0;
    for (int k = 0; k < K; k++) {
        res += a[j * K + k] * b[k * N + i];
    }
    c[j * N + i] = res;
}

__device__ float mm_readA(const float* a, int row, int col, int dimInner) {
    return a[row * dimInner + col];
}

__device__ float mm_readB(const float* b, int row, int col, int dimBOuter) {
    return b[row * dimBOuter + col];
}

__device__ void mm_write(float* c, int row, int col, float value, int dimBOuter) {
    c[row * dimBOuter + col] = value;
}

__global__ void matmulSharedKernel(float* c, const float* a, const float* b)
{    
    /*int3 localId = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
    int3 globalId = make_int3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);*/
    __shared__ float mm_Asub[tileAOuter][tileInner];
    __shared__ float mm_Bsub[tileInner][tileBOuter];

    int tileRow = threadIdx.y * workPerThread1;
    int tileCol = threadIdx.x * workPerThread0;

    int globalRow = (blockIdx.y * blockDim.y + threadIdx.y) * workPerThread1;
    int globalCol = (blockIdx.x * blockDim.x + threadIdx.x) * workPerThread0;

    int numTiles = (dimK - 1) / tileInner + 1;

    float acc[workPerThread1][workPerThread0];
    float ACached;
    float BCached[workPerThread0];

    for (int innerRow = 0; innerRow < workPerThread1; innerRow++) {
        for (int innerCol = 0; innerCol < workPerThread0; innerCol++) {
            acc[innerRow][innerCol] = 0.0;
        }
    }

    int ColPerThreadA = tileInner / workGroupSize0;
    int tileColA = threadIdx.x * ColPerThreadA;
    int RowPerThreadB = tileInner / workGroupSize1;
    int tileRowB = threadIdx.y * RowPerThreadB;

    for (int t = 0; t < numTiles; t++) {
        for (int innerRow = 0; innerRow < workPerThread1; innerRow++) {
            for (int innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
                int inputRow = tileRow + innerRow;
                int inputCol = tileColA + innerCol;

                mm_Asub[inputRow][inputCol] = mm_readA(a, globalRow + innerRow,
                    t * tileInner + inputCol, dimK);
            }
        }

        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
            for (int innerCol = 0; innerCol < workPerThread0; innerCol++) {
                int inputRow = tileRowB + innerRow;
                int inputCol = tileCol + innerCol;

                mm_Bsub[inputRow][inputCol] = mm_readB(b, t * tileInner + inputRow,
                    globalCol + innerCol, dimN);
            }
        }

        __syncthreads();

        for (int k = 0; k < tileInner; k++) {
            for (int inner = 0; inner < workPerThread0; inner++) {
                BCached[inner] = mm_Bsub[k][tileCol + inner];
            }

            for (int innerRow = 0; innerRow < workPerThread1; innerRow++) {
                ACached = mm_Asub[tileRow + innerRow][k];

                for (int innerCol = 0; innerCol < workPerThread0; innerCol++) {
                    acc[innerRow][innerCol] += ACached * BCached[innerCol];
                }
            }
        }

        __syncthreads();
    }

    for (int innerRow = 0; innerRow < workPerThread1; innerRow++) {
        for (int innerCol = 0; innerCol < workPerThread0; innerCol++) {
            if ((globalCol + innerCol) < dimN && (globalRow + innerRow) < dimM) {
                mm_write(c, globalRow + innerRow, globalCol + innerCol,
                    acc[innerRow][innerCol], dimN);
            }
        }
    }
}

__global__ void matmulSharedJJKernel(float* c, const float* a, const float* b)
{
    /*int3 localId = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
    int3 globalId = make_int3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);*/
    __shared__ float mm_Asub[tileAOuter][tileInner];
    __shared__ float mm_Bsub[tileInner][tileBOuter];

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    int globalRow = blockIdx.y * tileAOuter;
    int globalCol = blockIdx.x * tileBOuter;

    int numTiles = (dimK - 1) / tileInner + 1;

    float acc[workPerThread1][workPerThread0];
    float ACached;
    float BCached[workPerThread0];

    for (int innerRow = 0; innerRow < workPerThread1; innerRow++) {
        for (int innerCol = 0; innerCol < workPerThread0; innerCol++) {
            acc[innerRow][innerCol] = 0.0;
        }
    }

    for (int t = 0; t < numTiles; t++) {
        for (int inputRow = localRow; inputRow < tileAOuter; inputRow += workGroupSize1) {
            for (int inputCol = localCol; inputCol < tileInner; inputCol += workGroupSize0) {
                mm_Asub[inputRow][inputCol] = mm_readA(a, globalRow + inputRow,
                    t * tileInner + inputCol, dimK);
            }
        }

        for (int inputRow = localRow; inputRow < tileInner; inputRow += workGroupSize1) {
            for (int inputCol = localCol; inputCol < tileBOuter; inputCol +=workGroupSize0) {
                mm_Bsub[inputRow][inputCol] = mm_readB(b, t * tileInner + inputRow,
                    globalCol + inputCol, dimN);
            }
        }

        __syncthreads();

        for (int k = 0; k < tileInner; k++) {
            for (int inner = 0; inner < workPerThread0; inner++) {
                BCached[inner] = mm_Bsub[k][localCol + inner * workGroupSize0];
            }

            for (int innerRow = 0; innerRow < workPerThread1; innerRow++) {
                ACached = mm_Asub[localRow + innerRow * workGroupSize1][k];

                for (int innerCol = 0; innerCol < workPerThread0; innerCol++) {
                    acc[innerRow][innerCol] += ACached * BCached[innerCol];
                }
            }
        }

        __syncthreads();
    }

    for (int innerRow = 0; innerRow < workPerThread1; innerRow++) {
        for (int innerCol = 0; innerCol < workPerThread0; innerCol++) {
            int gRow = globalRow + localRow + innerRow * workGroupSize1;
            int gCol = globalCol + localCol + innerCol * workGroupSize0;
            if (gCol < dimN && gRow < dimM) {
                mm_write(c, gRow, gCol, acc[innerRow][innerCol], dimN);
            }
        }
    }
}

int main()
{
    float *a = new float[dimM * dimK];
    for (int i = 0; i < dimM; i++) {
        for (int j = 0; j < dimK; j++) {
            a[i * dimK + j] = (float)(rand()) / ((float)(RAND_MAX));
        }
    }
    float *b = new float[dimK * dimN];
    for (int i = 0; i < dimK; i++) {
        for (int j = 0; j < dimN; j++) {
            b[i * dimN + j] = (float)(rand()) / ((float)(RAND_MAX));
        }
    }
    float *expected = new float[dimM * dimN];
    for (int i = 0; i < dimM; i++) {
        for (int j = 0; j < dimN; j++) {
            float res = 0.0;
            for (int k = 0; k < dimK; k++) {
                res += a[i * dimK + k] * b[k * dimN + j];
            }
            expected[i * dimN + j] = res;
        }
    }

    float *c = new float[dimM * dimN];

    // Add vectors in parallel.
    cudaError_t cudaStatus = matmulWithCuda(c, a, b, dimM, dimK, dimN);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    
    int count = 0;
    for (int i = 0; i < dimM; i++) {
        for (int j = 0; j < dimN; j++) {
            if (expected[i * dimN + j] != c[i * dimN + j]) {
                printf("%f ", c[i * dimN + j]);
                count++;
                if (count == 8) {
                    count = 0;
                    printf("\n ");
                }
            }
        }
    }

    printf("\n");
    printf("\n");
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] expected;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t matmulWithCuda(float *c, const float *a, const float *b, int M, int K, int N)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, M * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, M * K * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, K * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(1, 1, 1);
    blockDim.x = workGroupSize0;
    blockDim.y = workGroupSize1;
    gridDim.x = (N - 1) / (blockDim.x * workPerThread0) + 1;
    gridDim.y = (M - 1) / (blockDim.y * workPerThread1) + 1;

    // Launch a kernel on the GPU with one thread for each element.
    matmulSharedKernel <<<gridDim, blockDim>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matmulNaiveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matmulKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
