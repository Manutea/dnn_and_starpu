#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cudnn.h"

#define REQUESTED_ALGO 10

cudnnFilterDescriptor_t filt_desc; 
cudnnPoolingDescriptor_t pool_desc;
cudnnConvolutionDescriptor_t conv_desc;
cudnnTensorDescriptor_t in_desc, out_desc;

cudnnHandle_t cudnn[STARPU_NMAXWORKERS];

struct convolution_params
{
  float alpha, beta;  
};

struct tensor 
{
  starpu_data_handle_t handle;
  int x, y, z, w;
};

void free_dnn(const struct convolution_params *);
void show_result(const int, starpu_data_handle_t);

void init_cudnn(void *);
void free_tensor(struct tensor);
struct tensor init_tensor(const float *, const int, const int, const int, const int);

void convolution_forward(void **, void *);
struct tensor submit_convolution_forward(int, int, int, int, int, int, struct tensor, struct tensor, struct convolution_params *);
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
struct tensor submit_max_pooling_forward(int, int, int, int, int, int, struct tensor, struct convolution_params *);
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

  struct convolution_params conv_params = {1.0, 0.0}; 

  const int in_n = 1, in_c = 1;
  const int in_size = in_n * in_c * in_h * in_w;          
  float *in_data = malloc(in_size * sizeof(float)); //starpu_malloc, ca evite de devoir faire memory_pin
  for(int i=0; i<in_size; i++) 
  {
    in_data[i] = i;
  }

  const int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
  const int filt_size = filt_k * filt_c * filt_h * filt_w;  
  float *filt_data = malloc(filt_size * sizeof(float)); //pareil
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

    struct tensor out = submit_convolution_forward(1, 1, 1, 1, 1, 1, in, filter, &conv_params);
    free_tensor(in);
    struct tensor out2 = submit_convolution_forward(1, 1, 1, 1, 1, 1, out, filter, &conv_params);
    free_tensor(out);
    struct tensor out3 = submit_max_pooling_forward(3, 3, 0, 0, 1, 1, in, &conv_params);    
    free_tensor(out2);

    if(show)
    {
      const int size = out3.x * out3.y * out3.z * out3.w;
      show_result(size, out3.handle);
    }

    free_tensor(out3);
    free_tensor(filter);

    starpu_memory_unpin(in_data, sizeof(in_data[0])*in_size);
    starpu_memory_unpin(filt_data, sizeof(filt_data[0])*filt_size);
  }

  free_dnn(&conv_params);
  starpu_shutdown();

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

  starpu_memory_pin(data, sizeof(data[0]) * size);
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
}

struct tensor submit_convolution_forward(int pad_h, int pad_w, int u, int v, int dil_h, int dil_w, struct tensor in, struct tensor filter, struct convolution_params *prms)
{
  //In Descriptor
  cudnnCreateTensorDescriptor(&in_desc);
  cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in.x, in.y, in.z, in.w);

  //Filter Descriptor
  cudnnCreateFilterDescriptor(&filt_desc);
  cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter.x, filter.y, filter.z, filter.w);

  //Convolution
  cudnnCreateConvolutionDescriptor(&conv_desc);
  cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, u, v, dil_h, dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);


  int out_n, out_c, out_h, out_w;
  cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w);

  //Tensor out
  starpu_data_handle_t out_handle;
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);
  starpu_vector_data_register(&out_handle, -1, NULL, out_n * out_c * out_h * out_w, sizeof(float));

  struct starpu_task *task = starpu_task_create();
  task->cl = &convolution_forward_cl;
  task->handles[0] = in.handle;
  task->handles[1] = filter.handle;
  task->handles[2] = out_handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct convolution_params);
  task->destroy = 0;

  int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  struct tensor out = {out_handle, out_n, out_c, out_h, out_w};
  return out;
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

struct tensor submit_max_pooling_forward(int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride, struct tensor in, struct convolution_params *prms) 
{
  //In Descriptor
  cudnnCreateTensorDescriptor(&in_desc);
  cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in.x, in.y, in.z, in.w);

  //Max Pooling
  cudnnCreatePoolingDescriptor(&pool_desc);
  cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

  int out_n, out_c, out_h, out_w;
  cudnnGetPooling2dForwardOutputDim(pool_desc, in_desc, &out_n, &out_c, &out_h, &out_w);

  //Tensor out
  starpu_data_handle_t out_handle;
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);
  starpu_vector_data_register(&out_handle, -1, NULL, out_n * out_c * out_h * out_w, sizeof(float));

  struct starpu_task *task = starpu_task_create();
  task->cl = &pooling_forward_cl;
  task->handles[0] = in.handle;
  task->handles[1] = out_handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct convolution_params);
  task->destroy = 0;

  int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
 
  struct tensor out = {out_handle, out_n, out_c, out_h, out_w};
  return out;
}
