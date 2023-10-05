// nvcc -arch=sm_70 -std=c++17 -c wrappers.cu -o wrappers.o
// nvcc -arch=sm_70 -std=c++17 -lnvidia-ml wrappers.o ./benchmark.cu -o benchmark && ./benchmark 25


#include "../../../curves/bls12_381/curve_config.cuh"
using namespace BLS12_381;

#include <iostream>
#include <fstream>
#include <nvml.h>
#include <unistd.h>

using namespace std;


// msm function header here, body defined in wrappers.cu
void msm_wrapper(
    scalar_t* scalars_d,
    affine_t* points_d,
    projective_t* results_d,
    unsigned size,
    unsigned large_bucket_factor=1,
    cudaStream_t stream=0
);


#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << endl;
        cerr << cudaGetErrorString(err) << endl;
    }
}

void genData(unsigned N) {
    // one affine point takes 96 bytes, one scalar takes 32 bytes
    ofstream file_obj;
    string fname = "./input" + to_string(N) + ".dat";
    file_obj.open(fname, ios::app);
    scalar_t* scalars = new scalar_t[N];
    affine_t* points = new affine_t[N];
    for (unsigned i = 0; i < N; i++) {
        scalars[i] = scalar_t::rand_host();
        points[i] = projective_t::to_affine(projective_t::rand_host());
        file_obj.write((char*)&scalars[i], sizeof(scalars[i]));
        file_obj.write((char*)&points[i], sizeof(points[i]));
    }
    file_obj.close();
}

tuple<scalar_t*, affine_t*> readData(unsigned N) {
    scalar_t* scalars = new scalar_t[N];
    affine_t* points = new affine_t[N];
    ifstream file_obj;
    string fname = "./input" + to_string(N) + ".dat";
    file_obj.open(fname, ios::in);
    for (unsigned i = 0; i < N; i++) {
        file_obj.read((char*)&scalars[i], sizeof(scalars[i]));
        file_obj.read((char*)&points[i], sizeof(points[i]));
    }
    return make_tuple(scalars,points);
}

int main(int argc, char *argv[])
{
    unsigned int batch_size = 1;
    unsigned int large_bucket_factor = 20; // FIXME: runtime errors for some values of this hyperparam
    unsigned int log_msm_size = atoi(argv[1]);
    unsigned int msm_size = 1 << log_msm_size;
    unsigned N = batch_size * msm_size;

    scalar_t* scalars = new scalar_t[N];
    affine_t* points = new affine_t[N];
    projective_t* out = new projective_t[1];
    scalar_t* scalars_d;
    affine_t* points_d;
    projective_t* out_d;

    printf("MSM size = %d...\n", msm_size);

    printf("Generating input scalars and points...\n");
    //genData(100000);
    scalar_t* sample_scalars = new scalar_t[N];
    affine_t* sample_points = new affine_t[N];
    // pseudo-random hack for speedup
    unsigned N_PRESAMPLED = 100000;
    tie(sample_scalars,sample_points) = readData(N_PRESAMPLED); 
    for (unsigned i = 0; i < N; i++) {
        scalars[i] = sample_scalars[i%N_PRESAMPLED];
        points[i] = sample_points[i%N_PRESAMPLED]; 

    };
    // copy data to device and warm-up
    cudaMalloc(&scalars_d, sizeof(scalar_t) * N);
    cudaMalloc(&points_d, sizeof(affine_t) * N);
    cudaMalloc(&out_d, sizeof(projective_t));
    cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(points_d, points, sizeof(affine_t) * N, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    for (int i = 0; i < 3; i++) {
        msm_wrapper(scalars_d,points_d,out_d, N, large_bucket_factor, 0);
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
    }

    printf("Computing MSM...\n");

    // initialize NVML objects for profiling
    nvmlReturn_t status;
    nvmlDevice_t device_handle;
    unsigned long long energy_start, energy_end;
    long long energy_total;
    cudaEvent_t time_start, time_stop;
    float time_total;
    unsigned int memClock;
    unsigned int ClockFreqNumber=200;
    unsigned int ClockFreqs[200];
    nvmlInit();
    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);
    nvmlDeviceGetHandleByIndex(0, &device_handle);
    nvmlDeviceGetApplicationsClock(device_handle, NVML_CLOCK_MEM, &memClock);
    printf("Memory clock is %d\n",memClock);
    nvmlDeviceGetSupportedGraphicsClocks(device_handle, memClock, &ClockFreqNumber, ClockFreqs );

    // run and profile code on device
    for (int i = 0; i < ClockFreqNumber; i++) {
        unsigned int freq = ClockFreqs[i];
        status = nvmlDeviceSetApplicationsClocks(device_handle, memClock, freq);
        usleep(100);
        for (int i = 0; i < 5; i++) {
            // start profiling
            cudaEventRecord(time_start, 0);
            nvmlDeviceGetTotalEnergyConsumption(device_handle,&energy_start);
            // profiled code
            msm_wrapper(scalars_d,points_d,out_d, N, large_bucket_factor, 0);
            cudaDeviceSynchronize();
            // end profiling
            cudaEventRecord(time_stop, 0);
            cudaEventSynchronize(time_stop);
            cudaEventElapsedTime(&time_total, time_start, time_stop);
            nvmlDeviceGetTotalEnergyConsumption(device_handle,&energy_end);
            energy_total=energy_end-energy_start;
            CHECK_LAST_CUDA_ERROR();
            printf("Param=%d,Freq=%d,Time=%f [ms],Energy=%lld [mJ]\n", log_msm_size, freq,time_total, energy_total);
        };
    };


    nvmlDeviceResetApplicationsClocks(device_handle);
    nvmlShutdown();
    //cudaMemcpy(out, out_d, sizeof(projective_t), cudaMemcpyDeviceToHost);
    //std::cout << projective_t::to_affine(out[0]) << std::endl;
}