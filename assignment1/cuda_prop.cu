#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int getCoresPerSM(int major, int minor) {
    // Defines cores per SM based on architecture generation
    switch (major) {
 case 2: // Fermi
            return (minor == 1) ? 48 : 32;
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            if (minor == 1 || minor == 2) return 128;
            if (minor == 0) return 64;
            return 128; // Default fallback for Pascal
        case 7: // Volta (7.0), Turing (7.5)
            return 64;
        case 8: // Ampere (8.0, 8.6, 8.7), Ada Lovelace (8.9)
            if (minor == 0) return 64;
            if (minor == 6 || minor == 9) return 128;
            return 64; // Default fallback for Ampere variants
        case 9: // Hopper (9.0), Blackwell (9.5)
            return 128;
        default:
            return 128; // Standard fallback for future architectures
    }
}

int main() {
    int deviceCount = 0;
    int driverVersion = 0;
    int runtimeVersion = 0;

    cudaError_t error = cudaDriverGetVersion(&driverVersion);
    if (error != cudaSuccess) {
        cerr << "CUDA Error (driver version): " << cudaGetErrorString(error) << endl;
        return 1;
    }

    error = cudaRuntimeGetVersion(&runtimeVersion);
    if (error != cudaSuccess) {
        cerr << "CUDA Error (runtime version): " << cudaGetErrorString(error) << endl;
        return 1;
    }

    cout << "CUDA driver version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << endl;
    cout << "CUDA runtime version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << endl;
    cout << endl;

    error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(error) << endl;
        return 1;
    }

    cout << "Found " << deviceCount << " CUDA device(s)." << endl << endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        if (error != cudaSuccess) {
            cerr << "CUDA Error (device properties): " << cudaGetErrorString(error) << endl;
            return 1;
        }

        int coresPerSM = getCoresPerSM(prop.major, prop.minor);
        int totalCores = prop.multiProcessorCount * coresPerSM;

        cout << "--- Device " << i << ": " << prop.name << " ---" << endl;
        cout << "  Compute Capability:          " << prop.major << "." << prop.minor << endl;
        cout << "  Total Global Memory:         " << prop.totalGlobalMem / (1024*1024) << " MB" << endl;
        cout << "  Streaming Multiprocessors:   " << prop.multiProcessorCount << endl;
        cout << "  CUDA Cores per SM:           " << coresPerSM << endl;
        cout << "  Total CUDA Cores:            " << totalCores << endl;
        cout << "  Max Threads Per Block:       " << prop.maxThreadsPerBlock << endl;
        cout << "  Shared Memory Per Block:     " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
        cout << "  Shared Memory Per SM:        " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << endl;
        cout << "  Warp Size:                   " << prop.warpSize << endl;
        cout << "  Registers Per Block:         " << prop.regsPerBlock << endl;
        cout << "  L2 Cache Size:               " << prop.l2CacheSize / 1024 << " KB" << endl;
        cout << "  Memory Clock Rate:           " << prop.memoryClockRate / 1000.0 << " MHz" << endl;
        cout << "  GPU Clock Rate:              " << prop.clockRate / 1000.0 << " MHz" << endl;
        cout << "  Memory Bus Width:            " << prop.memoryBusWidth << " bits" << endl;
        cout << "  Max Threads Per SM:          " << prop.maxThreadsPerMultiProcessor << endl;
        cout << "  Max Threads Dim:             (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl;
        cout << "  Max Grid Size:               (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
        cout << "  Concurrent Kernels:          " << (prop.concurrentKernels ? "Yes" : "No") << endl;
        cout << "  Async Engine Count:          " << prop.asyncEngineCount << endl;
        cout << "  Integrated GPU:              " << (prop.integrated ? "Yes" : "No") << endl;
        cout << "  Unified Addressing:          " << (prop.unifiedAddressing ? "Yes" : "No") << endl;
        cout << "  Can Map Host Memory:         " << (prop.canMapHostMemory ? "Yes" : "No") << endl;
        cout << "  ECC Enabled:                 " << (prop.ECCEnabled ? "Yes" : "No") << endl;
        cout << endl;
    }

    return 0;
}



/**
Output : 

CUDA driver version: 13.2
CUDA runtime version: 12.9

Found 1 CUDA device(s).

--- Device 0: NVIDIA GeForce GTX 1650 ---
  Compute Capability:          7.5
  Total Global Memory:         3717 MB
  Streaming Multiprocessors:   14
  CUDA Cores per SM:           64
  Total CUDA Cores:            896
  Max Threads Per Block:       1024
  Shared Memory Per Block:     48 KB
  Shared Memory Per SM:        64 KB
  Warp Size:                   32
  Registers Per Block:         65536
  L2 Cache Size:               1024 KB
  Memory Clock Rate:           6001 MHz
  GPU Clock Rate:              1515 MHz
  Memory Bus Width:            128 bits
  Max Threads Per SM:          1024
  Max Threads Dim:             (1024, 1024, 64)
  Max Grid Size:               (2147483647, 65535, 65535)
  Concurrent Kernels:          Yes
  Async Engine Count:          3
  Integrated GPU:              No
  Unified Addressing:          Yes
  Can Map Host Memory:         Yes
  ECC Enabled:                 No

*/
