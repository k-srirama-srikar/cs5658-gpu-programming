#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__device__ int arrival_count = 0;
__device__ int departure_count = 0;
__device__ int release_gate = 0;

__device__ void global_barrier(int num_blocks){
    __syncthreads();

    if (threadIdx.x == 0){
        int arrived = atomicAdd(&arrival_count, 1);

        if (arrived == num_blocks - 1){
            arrival_count = 0;
            __threadfence();
            release_gate = 1;
        }
    }

    while (atomicAdd(&release_gate, 0) == 0){
        // wait until all blocks have reached the barrier
    }

    __syncthreads();

    if(threadIdx.x == 0){
        int left = atomicAdd(&departure_count, 1);

        if (left == num_blocks - 1) {
            departure_count = 0;
            __threadfence();
            release_gate = 0;
        }
    }

    while (atomicAdd(&release_gate, 0) == 1) {
        // wait until all blocks have cleared the barrier
    }

    __syncthreads();
}

__global__ void barrier_test_kernel(int num_blocks){
    if (threadIdx.x == 0) {
        printf("Block %d: before barrier\n", blockIdx.x);
    }

    global_barrier(num_blocks);

    if (threadIdx.x == 0) {
        printf("Block %d: after barrier\n", blockIdx.x);
    }
}

void check_cuda_error(const char *message){
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << message << ": " << cudaGetErrorString(error) << endl;
        exit(1);
    }
}

int main(){
    const int blocks = 4;
    const int threads_per_block = 256;

    cout << "\nLaunching kernel\n" << endl;

    barrier_test_kernel<<<blocks, threads_per_block>>>(blocks);

    check_cuda_error("kernel launch");

    cudaDeviceSynchronize();

    check_cuda_error("kernel execution");

    cout << "\nKernel completed" << endl;

    return 0;
}


/**
Output : 

Launching kernel

Block 0: before barrier
Block 2: before barrier
Block 3: before barrier
Block 1: before barrier
Block 0: after barrier
Block 2: after barrier
Block 3: after barrier
Block 1: after barrier

Kernel completed

*/
