#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cudnn.h"

#define REQUESTED_ALGO 10

cudnnHandle_t cudnn[STARPU_NMAXWORKERS];

struct tensor 
{
  starpu_data_handle_t handle;
  int x, y, z, w;
};

void free_dnn();
void show_result(const int, starpu_data_handle_t);

void init_cudnn(void *);
void free_tensor(struct tensor);
struct tensor init_tensor(const float *, const int, const int, const int, const int);

struct convolution_forward_params
{
  float alpha, beta;
  cudnnTensorDescriptor_t in_desc, out_desc;
  cudnnFilterDescriptor_t filt_desc;
  cudnnConvolutionDescriptor_t conv_desc;
};
void convolution_forward(void **, void *);
struct tensor submit_convolution_forward(int, int, int, int, int, int, float, float, struct tensor, struct tensor);
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

struct pooling_forward_params
{
  float alpha, beta;
  cudnnTensorDescriptor_t in_desc, out_desc;
  cudnnPoolingDescriptor_t pool_desc;
};
void pooling_forward(void **, void *);
struct tensor submit_max_pooling_forward(int, int, int, int, int, int, float, float, struct tensor);
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

  const int show = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int in_h = atoi(argv[3]);
  const int in_w = atoi(argv[4]);

  const int in_n = 1, in_c = 1;
  const int in_size = in_n * in_c * in_h * in_w;          
  float *in_data;
  starpu_malloc((void**)&in_data, in_size * sizeof(float));
  for(int i=0; i<in_size; i++) 
  {
    in_data[i] = i;
  }

  const int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
  const int filt_size = filt_k * filt_c * filt_h * filt_w;  
  float *filt_data;
  starpu_malloc((void**)&filt_data, filt_size * sizeof(float));
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
  starpu_execute_on_each_worker(init_cudnn, cudnn, STARPU_CUDA);

  for(int i=0; i<repeat; i++)
  {
    struct tensor filter = init_tensor(filt_data, filt_k, filt_c, filt_h, filt_w);
    struct tensor in = init_tensor(in_data, in_n, in_c, in_h, in_w);

    struct tensor out = submit_convolution_forward(1, 1, 1, 1, 1, 1, 1.0, 0.0, in, filter);
    free_tensor(in);
    struct tensor out2 = submit_convolution_forward(1, 1, 1, 1, 1, 1, 1.0, 0.0, out, filter);
    free_tensor(out);
    struct tensor out3 = submit_max_pooling_forward(3, 3, 0, 0, 1, 1, 1.0, 0.0, out2);    
    free_tensor(out2);

    if(show)
    {
      const int size = out3.x * out3.y * out3.z * out3.w;
      show_result(size, out3.handle);
    }

    free_tensor(out3);
    free_tensor(filter);
  }

  starpu_free(in_data);
  starpu_free(filt_data);

  free_dnn();
  starpu_shutdown();

  return 0;
}

void free_dnn() 
{
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


//------- Initialize --------
void init_cudnn(void *arg) 
{
  cudnnHandle_t *cudnn_ = (cudnnHandle_t *) arg;
  const int id = starpu_worker_get_id();
  cudnnCreate(&cudnn_[id]);
  cudnnSetStream(cudnn_[id], starpu_cuda_get_local_stream());
}

void free_tensor(struct tensor tensor)
{
  starpu_data_unregister_submit(tensor.handle);
}

struct tensor init_tensor(const float *data, const int x, const int y, const int z, const int w)
{
  starpu_data_handle_t handle;
  cudnnFilterDescriptor_t desc; 

  cudnnCreateFilterDescriptor(&desc);
  cudnnSetFilter4dDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x, y, z, w);
  const int size = x * y * z * w;
  cudnnDestroyTensorDescriptor(desc);

  starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)data, size, sizeof(data[0]));
  struct tensor filter_tensor = {handle, x, y, z, w};

  return filter_tensor;
}


//------- CONVOLUTION --------
void convolution_forward(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  const float *filt_data  = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[2]); 
  const struct convolution_forward_params *prms = (struct convolution_forward_params *)_args;
  const int id = starpu_worker_get_id();

  //This function attempts all algorithms available for cudnnConvolutionForward().
  int n_returnedAlgo;  
  cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf[REQUESTED_ALGO];
  cudnnFindConvolutionForwardAlgorithm(cudnn[id], prms->in_desc, prms->filt_desc, prms->conv_desc, prms->out_desc, REQUESTED_ALGO, &n_returnedAlgo, fwd_algo_perf);

  //This function returns the amount of GPU memory workspace the user needs to allocate to be able to call cudnnConvolutionForward() with the specified algorithm.
  int ws_size = 0;
  cudnnConvolutionFwdAlgo_t fwd_algo = fwd_algo_perf[0].algo;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn[id], prms->in_desc, prms->filt_desc, prms->conv_desc, prms->out_desc, fwd_algo, &ws_size);

  if(ws_size > 0)
  {
    float *ws_data;
    cudaMalloc(&ws_data, ws_size);
    cudnnConvolutionForward(cudnn[id], &prms->alpha, prms->in_desc, in_data, prms->filt_desc, filt_data, prms->conv_desc, 
                           fwd_algo, ws_data, ws_size, &prms->beta, prms->out_desc, out_data);
    cudaFree(ws_data);
  }
  else 
  {
    cudnnConvolutionForward(cudnn[id], &prms->alpha, prms->in_desc, in_data, prms->filt_desc, filt_data, prms->conv_desc, 
                           fwd_algo, NULL, ws_size, &prms->beta, prms->out_desc, out_data);
  }
}

struct tensor submit_convolution_forward(int pad_h, int pad_w, int u, int v, int dil_h, int dil_w, float alpha, float beta, struct tensor in, struct tensor filter)
{
  struct convolution_forward_params prms;
  prms.alpha = alpha;
  prms.beta = beta;

  //In Descriptor
  cudnnCreateTensorDescriptor(&prms.in_desc);
  cudnnSetTensor4dDescriptor(prms.in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in.x, in.y, in.z, in.w);

  //Filter Descriptor
  cudnnCreateFilterDescriptor(&prms.filt_desc);
  cudnnSetFilter4dDescriptor(prms.filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter.x, filter.y, filter.z, filter.w);

  //Convolution
  cudnnCreateConvolutionDescriptor(&prms.conv_desc);
  cudnnSetConvolution2dDescriptor(prms.conv_desc, pad_h, pad_w, u, v, dil_h, dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

  int out_n, out_c, out_h, out_w;
  cudnnGetConvolution2dForwardOutputDim(prms.conv_desc, prms.in_desc, prms.filt_desc, &out_n, &out_c, &out_h, &out_w);

  //Tensor out
  starpu_data_handle_t out_handle;
  cudnnCreateTensorDescriptor(&prms.out_desc);
  cudnnSetTensor4dDescriptor(prms.out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);
  starpu_vector_data_register(&out_handle, -1, NULL, out_n * out_c * out_h * out_w, sizeof(float));

  struct starpu_task *task = starpu_task_create();
  task->cl = &convolution_forward_cl;
  task->handles[0] = in.handle;
  task->handles[1] = filter.handle;
  task->handles[2] = out_handle;
  task->cl_arg = &prms;
  task->cl_arg_size = sizeof(struct convolution_forward_params);
  task->destroy = 0;

  int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  //cudnnDestroyTensorDescriptor(prms.in_desc);
  //cudnnDestroyFilterDescriptor(prms.filt_desc);
  //cudnnDestroyConvolutionDescriptor(prms.conv_desc);
  //cudnnDestroyTensorDescriptor(prms.out_desc);

  struct tensor out = {out_handle, out_n, out_c, out_h, out_w};
  return out;
}


//------- POOLING --------
void pooling_forward(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[1]); 
  const struct pooling_forward_params *prms = (struct pooling_forward_params *)_args;
  const int id = starpu_worker_get_id();

  cudnnPoolingForward(cudnn[id], prms->pool_desc, &prms->alpha, prms->in_desc, in_data, &prms->beta, prms->out_desc, out_data);
}

struct tensor submit_max_pooling_forward(int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride, float alpha, float beta, struct tensor in) 
{
  struct pooling_forward_params prms;
  prms.alpha = alpha;
  prms.beta = beta;

  //In Descriptor
  cudnnCreateTensorDescriptor(&prms.in_desc);
  cudnnSetTensor4dDescriptor(prms.in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in.x, in.y, in.z, in.w);

  //Max Pooling
  cudnnCreatePoolingDescriptor(&prms.pool_desc);
  cudnnSetPooling2dDescriptor(prms.pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

  int out_n, out_c, out_h, out_w;
  cudnnGetPooling2dForwardOutputDim(prms.pool_desc, prms.in_desc, &out_n, &out_c, &out_h, &out_w);

  //Tensor out
  starpu_data_handle_t out_handle;
  cudnnCreateTensorDescriptor(&prms.out_desc);
  cudnnSetTensor4dDescriptor(prms.out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);
  starpu_vector_data_register(&out_handle, -1, NULL, out_n * out_c * out_h * out_w, sizeof(float));

  struct starpu_task *task = starpu_task_create();
  task->cl = &pooling_forward_cl;
  task->handles[0] = in.handle;
  task->handles[1] = out_handle;
  task->cl_arg = &prms;
  task->cl_arg_size = sizeof(struct pooling_forward_params);
  task->destroy = 0;

  int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
 
  //cudnnDestroyTensorDescriptor(in_desc);
  //cudnnDestroyTensorDescriptor(out_desc);
  //cudnnDestroyPoolingDescriptor(pool_desc);

  struct tensor out = {out_handle, out_n, out_c, out_h, out_w};
  return out;
}
