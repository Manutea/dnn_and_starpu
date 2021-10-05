#include <starpu.h>

static __global__ void cuda_dev_const(float *px, float k)
{
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        px[tid] = k;
}

static __global__ void cuda_dev_iota(float *px)
{
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        px[tid] = tid;
}

extern "C" void dev_const(void *descr[], void *_args)
{
        float *filt_data = (float *)STARPU_MATRIX_GET_PTR(descr[0]);

        int filt_size = STARPU_MATRIX_GET_NX(descr[0]);
        int filt_nb = STARPU_MATRIX_GET_NY(descr[0]);

        cuda_dev_const<<<filt_size, filt_nb, 0, starpu_cuda_get_local_stream()>>>(filt_data, 1.f);

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
        cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

extern "C" void dev_iota(void *descr[], void *_args)
{
        float *in_data = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
        int in_size = STARPU_MATRIX_GET_NX(descr[0]);
        int in_nb = STARPU_MATRIX_GET_NY(descr[0]);
 
        cuda_dev_iota<<<in_size, in_nb, 0, starpu_cuda_get_local_stream()>>>(in_data);

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
        cudaStreamSynchronize(starpu_cuda_get_local_stream());
}