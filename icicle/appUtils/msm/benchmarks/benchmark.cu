// nvcc -arch=sm_70 -std=c++17 -c wrappers.cu -o wrappers.o
// nvcc -arch=sm_70 -std=c++17 wrappers.o ./benchmark.cu -o benchmark && ./benchmark

#include "../../../curves/bls12_381/curve_config.cuh"
using namespace BLS12_381;

#include <iostream>
#include <fstream>
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


int main()
{
    printf("Starting benchmark...\n");
    unsigned batch_size = 1;
    unsigned large_bucket_factor = 20; // FIXME: runtime errors for some values of this hyperparam
    unsigned msm_size = 1<<25;
    msm_size = msm_size;
    unsigned N = batch_size * msm_size;

    scalar_t* scalars = new scalar_t[N];
    affine_t* points = new affine_t[N];
    projective_t* out = new projective_t[1];
    scalar_t* scalars_d;
    affine_t* points_d;
    projective_t* out_d;

    printf("MSM size = %d...\n", msm_size);

    printf("Host: Generating input scalars and points...\n");
    //genData(100000);
    scalar_t* sample_scalars = new scalar_t[N];
    affine_t* sample_points = new affine_t[N];
    // pseudo-random hack for speedup
    unsigned N_PRESAMPLED = 10000;
    tie(sample_scalars,sample_points) = readData(N_PRESAMPLED); 
    for (unsigned i = 0; i < N; i++) {
        scalars[i] = sample_scalars[i%N_PRESAMPLED];
        points[i] = sample_points[i%N_PRESAMPLED]; 

    };

    printf("Host: Computing MSM...\n");
    cudaMalloc(&scalars_d, sizeof(scalar_t) * N);
    cudaMalloc(&points_d, sizeof(affine_t) * N);
    cudaMalloc(&out_d, sizeof(projective_t));
    cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(points_d, points, sizeof(affine_t) * N, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    msm_wrapper(scalars_d,points_d,out_d, N, large_bucket_factor, 0);;
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    //cudaMemcpy(out, out_d, sizeof(projective_t), cudaMemcpyDeviceToHost);
    //std::cout << projective_t::to_affine(out[0]) << std::endl;
}