// wrappers.cu
#include "../../../curves/bls12_381/curve_config.cuh"
#include "../msm.cu"

using namespace BLS12_381;

void msm_wrapper(
    scalar_t* scalars_d,
    affine_t* points_d,
    projective_t* results_d,
    unsigned size,
    unsigned large_bucket_factor=10,
    cudaStream_t stream=0
) {
    large_msm<scalar_t, projective_t, affine_t>(
        scalars_d,
        points_d,
        size,
        results_d,
        true,
        false,
        large_bucket_factor,
        stream
    );
}