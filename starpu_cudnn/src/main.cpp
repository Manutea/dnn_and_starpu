#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cudnn.h"

#define REQUESTED_ALGO 10

int toto=0;

double sumCudaTime = 0.0;
double sumDelayTime = 0.0;
double sumLengthTime = 0.0;
double sumCudnnTime = 0.0;

cudnnFilterDescriptor_t filt_desc; 
cudnnPoolingDescriptor_t pool_desc;
cudnnConvolutionDescriptor_t conv_desc;
cudnnTensorDescriptor_t in_desc, out_desc;

cudnnHandle_t cudnn[STARPU_NMAXWORKERS];

struct convolution_params
{
  float alpha, beta;  
};

void init_cudnn(void *);
void show_result(const int, starpu_data_handle_t);

void convolution_forward(void **, void *);
void free_dnn(const struct convolution_params *);
starpu_data_handle_t init_filter(const float *, const int, const int, const int, const int, const struct convolution_params *);
starpu_data_handle_t submit_convolution_forward(int *, starpu_data_handle_t, starpu_data_handle_t, struct convolution_params *, struct starpu_task *);
static struct starpu_perfmodel convolution_forward_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "convolution_forward_model"
};
static struct starpu_codelet convolution_forward_cl =
{
  .cuda_funcs = {convolution_forward},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 3,
  .modes = {STARPU_R, STARPU_R, STARPU_W},
  .model = &convolution_forward_model,
};

void pooling_forward(void **, void *);
starpu_data_handle_t submit_pooling_forward(int, int, int, int, int, int, int *, starpu_data_handle_t, cudnnPoolingMode_t, cudnnNanPropagation_t, struct convolution_params *, struct starpu_task *);
static struct starpu_perfmodel pooling_forward_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "pooling_forward_model"
};
static struct starpu_codelet pooling_forward_cl =
{
  .cuda_funcs = {pooling_forward},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 2,
  .modes = {STARPU_R, STARPU_W},
  .model = &pooling_forward_model,
};

int main(int argc, char **argv)
{
  if(argc != 5)
  {
    printf("\nshow_result repeat in_h in_w\n");
    return -1;
  }

  auto test = 0;

  const int show = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int in_h = atoi(argv[3]);
  const int in_w = atoi(argv[4]);

  const int in_n = 1, in_c = 1;
  const int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
  const int pad_h = 1, pad_w = 1, str_h = 1, str_w = 1, dil_h = 1, dil_w = 1;
  struct convolution_params conv_params = {1.0, 0.0}; 

  const int in_size = in_n * in_c * in_h * in_w;          
  float *in_data = malloc(in_size * sizeof(float));
  for(int i=0; i<in_size; i++) 
  {
    in_data[i] = i;
  }

  const int filt_size = filt_k * filt_c * filt_h * filt_w;  
  float *filt_data = malloc(filt_size * sizeof(float));
  for(int i=0; i<filt_size; i++) 
  {
    filt_data[i] = 1.0f;
  }

  const int ret = starpu_init(NULL);
  if (ret == -ENODEV)
  {
    return 77;
  }

  /* Enable profiling */
  starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

  int gpuprocs[STARPU_NMAXWORKERS];
  const unsigned ngpus =  starpu_cuda_worker_get_count();
  starpu_worker_get_ids_by_type(STARPU_CUDA_WORKER, gpuprocs, ngpus);
  starpu_execute_on_each_worker(init_cudnn, cudnn, STARPU_CUDA);

  for(int i=0; i<repeat; i++)
  {
    struct starpu_task *task;
    starpu_data_handle_t filt_handle = init_filter(filt_data, filt_k, filt_c, filt_h, filt_w, &conv_params);

    //Convolution
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    //Tensor in
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w);
    const int in_size = in_n * in_c * in_h * in_w;
    starpu_memory_pin(in_data, sizeof(in_data[0]) * in_size);
    starpu_data_handle_t in_handle;
    starpu_vector_data_register(&in_handle, STARPU_MAIN_RAM, (uintptr_t)in_data, in_size, sizeof(in_data[0]));

    int out_size = 0;
    //starpu_data_handle_t out_handle = submit_convolution_forward(&out_size, in_handle, filt_handle, &conv_params, task);    
    //starpu_data_unregister_submit(in_handle);
    //printf("out size : %d\n", out_size);

    //cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 6, 6);
    //starpu_data_handle_t out_handle2 = submit_conv(&out_size, out_handle, filt_handle, &conv_params, task);
    //starpu_data_unregister_submit(out_handle);
    //printf("data size : %d\n", out_size);

    starpu_data_handle_t out_handle = submit_pooling_forward(3, 3, 0, 0, 1, 1, &out_size, in_handle, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, &conv_params, task);    
    starpu_data_unregister_submit(in_handle);
    printf("out size : %d\n", out_size);

    if(show)
    {
      show_result(out_size, out_handle);
    }

    starpu_data_unregister(out_handle);
    starpu_data_unregister(filt_handle);

    starpu_memory_unpin(in_data, sizeof(in_data[0])*in_size);
    starpu_memory_unpin(filt_data, sizeof(filt_data[0])*filt_size);
  }

  free_dnn(&conv_params);
  starpu_shutdown();

  printf("\n");
  printf("CUDA : %lf\n", sumCudaTime/(double)repeat);
  printf("CUDNN : %lf\n", sumCudnnTime/(double)repeat);
  printf("Delay : %lf\n", sumDelayTime/(double)repeat);
  printf("Length : %lf\n", sumLengthTime/(double)repeat);

  free(in_data);
  free(filt_data);

  return 0;
}

void free_dnn(const struct convolution_params *prms) 
{
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyFilterDescriptor(filt_desc);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(pool_desc);

  for(int i = 0; i < starpu_cuda_worker_get_count(); i++)
  {
    cudnnDestroy(cudnn[i]);
  }
}

void show_result(const int size, starpu_data_handle_t handle)
{
  starpu_data_acquire(handle, STARPU_R);
  float *data= starpu_data_get_local_ptr(handle);
  for(int i=0; i<size; i++) 
  {
    printf("%f \n", data[i]);
  }
  starpu_data_release(handle);
}


//------- Initialize CUDNN --------
void init_cudnn(void *arg) 
{
  cudnnHandle_t *cudnn_ = (cudnnHandle_t *) arg;
  const int id = starpu_worker_get_id();
  cudnnCreate(&cudnn_[id]);
  cudnnSetStream(cudnn_[id], starpu_cuda_get_local_stream());
}


//------- CONVOLUTION --------
void convolution_forward(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  const float *filt_data  = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[2]); 
  const struct convolution_params *prms = (struct convolution_params *)_args;
  const int id = starpu_worker_get_id();

  //This function attempts all algorithms available for cudnnConvolutionForward().
  int n_returnedAlgo;  
  cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf[REQUESTED_ALGO];
  cudnnFindConvolutionForwardAlgorithm(cudnn[id], in_desc, filt_desc, conv_desc, out_desc, REQUESTED_ALGO, &n_returnedAlgo, fwd_algo_perf);

  //This function returns the amount of GPU memory workspace the user needs to allocate to be able to call cudnnConvolutionForward() with the specified algorithm.
  int ws_size = 0;
  cudnnConvolutionFwdAlgo_t fwd_algo = fwd_algo_perf[0].algo;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn[id], in_desc, filt_desc, conv_desc, out_desc, fwd_algo, &ws_size);

  float elapsed=0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, starpu_cuda_get_local_stream());

  if(ws_size > 0)
  {
    float *ws_data;
    cudaMalloc(&ws_data, ws_size);
    cudnnConvolutionForward(cudnn[id], &prms->alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, 
                           fwd_algo, ws_data, ws_size, &prms->beta, out_desc, out_data);
    cudaFree(ws_data);
    printf("\n\n--------HEY !!!-------\n\n");
  }
  else 
  {
    cudnnConvolutionForward(cudnn[id], &prms->alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, 
                           fwd_algo, NULL, ws_size, &prms->beta, out_desc, out_data);
  }

  cudaEventRecord(stop, starpu_cuda_get_local_stream());
  cudaEventSynchronize (stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  sumCudaTime += elapsed;
  sumCudnnTime += fwd_algo_perf[0].time;
}

starpu_data_handle_t init_filter(const float *filter, const int k, const int c, const int h, const int w, const struct convolution_params *prms)
{
  starpu_data_handle_t filter_h;
  cudnnCreateFilterDescriptor(&filt_desc);
  cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w);
  const int filt_size = k * c * h * w;

  starpu_memory_pin(filter, sizeof(filter[0]) * filt_size);
  starpu_vector_data_register(&filter_h, STARPU_MAIN_RAM, (uintptr_t)filter, filt_size, sizeof(filter[0]));
  return filter_h;
}

starpu_data_handle_t submit_convolution_forward(int *out_size, starpu_data_handle_t in_handle, starpu_data_handle_t filt_handle, struct convolution_params *prms, struct starpu_task *task_r)
{
  //Setup the output tensor and allocate the proper amount of memory prior to launch the actual convolution
  int out_n, out_c, out_h, out_w;
  cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w);
  *out_size = out_n * out_c * out_h * out_w;

  //Tensor out
  starpu_data_handle_t out_handle;
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);
  starpu_vector_data_register(&out_handle, -1, NULL, out_size, sizeof(float));

  struct starpu_task *task = starpu_task_create();
  task->cl = &convolution_forward_cl;
  task->handles[0] = in_handle;
  task->handles[1] = filt_handle;
  task->handles[2] = out_handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct convolution_params);
  task->destroy = 0;

  if(task)
  {
    task_r = &task;
  }

  int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  starpu_task_wait_for_all();
  struct starpu_profiling_task_info *info = task->profiling_info;
  double delay = starpu_timing_timespec_delay_us(&info->submit_time, &info->start_time);
  double length = starpu_timing_timespec_delay_us(&info->start_time, &info->end_time);
  starpu_task_destroy(task);
  sumDelayTime += delay/1000.0;
  sumLengthTime += length/1000.0;

  return out_handle;
}


//------- POOLING --------
void pooling_forward(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[1]); 
  const struct convolution_params *prms = (struct convolution_params *)_args;
  const int id = starpu_worker_get_id();

  cudnnPoolingForward(cudnn[id], pool_desc, &prms->alpha, in_desc, in_data, &prms->beta, out_desc, out_data);

}

starpu_data_handle_t submit_pooling_forward(int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride,
int *out_size, starpu_data_handle_t in_handle, cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt, struct convolution_params *prms, struct starpu_task *task_r) 
{
  cudnnCreatePoolingDescriptor(&pool_desc);
  cudnnSetPooling2dDescriptor(pool_desc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
  //cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 3, 3, 0, 0, 1, 1);
  
  int out_n, out_c, out_h, out_w;
  cudnnGetPooling2dForwardOutputDim(pool_desc, in_desc, &out_n, &out_c, &out_h, &out_w);
  *out_size = out_n * out_c * out_h * out_w;

  //Tensor out
  starpu_data_handle_t out_handle;
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);
  starpu_vector_data_register(&out_handle, -1, NULL, out_size, sizeof(float));

  struct starpu_task *task = starpu_task_create();
  task->cl = &pooling_forward_cl;
  task->handles[0] = in_handle;
  task->handles[1] = out_handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct convolution_params);
  task->destroy = 0;

  if(task)
  {
    task_r = &task;
  }

  int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
  
  starpu_task_wait_for_all();
  struct starpu_profiling_task_info *info = task->profiling_info;
  double delay = starpu_timing_timespec_delay_us(&info->submit_time, &info->start_time);
  double length = starpu_timing_timespec_delay_us(&info->start_time, &info->end_time);
  starpu_task_destroy(task);
  sumDelayTime += delay/1000.0;
  sumLengthTime += length/1000.0;

  return out_handle;
}
