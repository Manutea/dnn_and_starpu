#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "cublas_v2.h"
#include "cudnn.h"

#define REQUESTED_ALGO 10

cudnnHandle_t cudnn[STARPU_NMAXWORKERS];

struct _tensor 
{
  starpu_data_handle_t handle;
  int n, c, h, w;
};
typedef struct _tensor *tensor;

void free_dnn();
void show_result(const tensor);

void init_cudnn(void *);
void free_tensor(tensor);
tensor init_tensor(const float *, const int, const int, const int, const int);

struct convolution2D_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
  int filter_n, filter_c, filter_h, filter_w;
  int pad_h, pad_w, u, v, dil_h, dil_w;
  int out_n, out_c, out_h, out_w;
};
void convolution2D_forward(void **, void *);
tensor submit_convolution2D_forward(const int, const int, const int, const int, const int, const int, 
                                        const float, const float, const tensor, const tensor, const tensor);
static struct starpu_perfmodel convolution2D_forward_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "convolution2D_forward_model"
};
static struct starpu_codelet convolution2D_forward_cl =
{
  .cuda_funcs = {convolution2D_forward},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 4,
  .modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_W},
  .model = &convolution2D_forward_model,
};

struct pooling2D_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
  cudnnPoolingMode_t mode;
  cudnnNanPropagation_t maxpoolingNanOpt;
  int windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride;
  int out_n, out_c, out_h, out_w;
};
void pooling2D_forward(void **, void *);
tensor submit_max_pooling2D_forward(const int, const int, const int, int, const int, const int, const float, const float, const tensor, const tensor);
static struct starpu_perfmodel pooling2D_forward_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "pooling2D_forward_model"
};
static struct starpu_codelet pooling2D_forward_cl =
{
  .cuda_funcs = {pooling2D_forward},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 3,
  .modes = {STARPU_R, STARPU_R, STARPU_W},
  .model = &pooling2D_forward_model,
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

  float *conv1_bias_data;
  starpu_malloc((void**)&conv1_bias_data, 1 * sizeof(float));
  conv1_bias_data[0] = 0.0f;

  float *conv2_bias_data;
  starpu_malloc((void**)&conv2_bias_data, 1 * sizeof(float));
  conv2_bias_data[0] = 0.0f;

  float *pool_bias_data;
  starpu_malloc((void**)&pool_bias_data, 1 * sizeof(float));
  pool_bias_data[0] = 0.0f;

  const int ret = starpu_init(NULL);
  if (ret == -ENODEV)
  {
    return 77;
  }

  /* Enable profiling */
  starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

  starpu_execute_on_each_worker(init_cudnn, cudnn, STARPU_CUDA);

  const tensor filter = init_tensor(filt_data, filt_k, filt_c, filt_h, filt_w);
  const tensor pool_bias = init_tensor(pool_bias_data, 1, 1, 1, 1);
  const tensor conv1_bias = init_tensor(conv1_bias_data, 1, 1, 1, 1);
  const tensor conv2_bias = init_tensor(conv2_bias_data, 1, 1, 1, 1);
  for(int i=0; i<repeat; i++)
  {
    const tensor in = init_tensor(in_data, in_n, in_c, in_h, in_w);

    const tensor out = submit_convolution2D_forward(1, 1, 1, 1, 1, 1, 1.0, 0.0, in, filter, conv1_bias);
    free_tensor(in);
    const tensor out2 = submit_convolution2D_forward(1, 1, 1, 1, 1, 1, 1.0, 0.0, out, filter, conv2_bias);
    free_tensor(out);
    const tensor out3 = submit_max_pooling2D_forward(3, 3, 0, 0, 1, 1, 1.0, 0.0, out2, pool_bias);    
    free_tensor(out2);

    if(show && i == repeat - 1)
    {
      show_result(out3);
    }

    free_tensor(out3);
  }

  free_tensor(filter);
  free_tensor(pool_bias);
  free_tensor(conv1_bias);
  free_tensor(conv2_bias);

  starpu_free(pool_bias_data);
  starpu_free(conv1_bias_data);
  starpu_free(conv2_bias_data);
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

void show_result(const tensor tensor)
{
  starpu_data_acquire(tensor->handle, STARPU_R);
  const float *data= starpu_data_get_local_ptr(tensor->handle);
  for(int i=0; i<tensor->n * tensor->c * tensor->h * tensor->w; i++) 
  {
    printf("%f \n", data[i]);
  }
  starpu_data_release(tensor->handle);
}


//------- Initialize --------
void init_cudnn(void *arg) 
{
  cudnnHandle_t *cudnn_ = (cudnnHandle_t *) arg;
  const int id = starpu_worker_get_id();
  cudnnCreate(&cudnn_[id]);
  cudnnSetStream(cudnn_[id], starpu_cuda_get_local_stream());
}

void free_tensor(tensor tensor)
{
  starpu_data_unregister_submit(tensor->handle);
  free(tensor);
}

tensor init_tensor(const float *data, const int n, const int c, const int h, const int w)
{
  tensor out = (tensor)malloc(sizeof(tensor));
  out->n = n;
  out->c = c;
  out->h = h;
  out->w = w;
  starpu_vector_data_register(&out->handle, STARPU_MAIN_RAM, (uintptr_t)data, n * c * h * w, sizeof(data[0]));
  return out;
}


//------- CONVOLUTION --------
void convolution2D_forward(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  const float *filt_data  = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
  float *bias_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[2]); 
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[3]); 
  const struct convolution2D_forward_params *prms = (const struct convolution2D_forward_params *)_args;
  const int id = starpu_worker_get_id();

  cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
  cudnnFilterDescriptor_t filt_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  //In Descriptor
  cudnnCreateTensorDescriptor(&in_desc);
  cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->in_n, prms->in_c, prms->in_h, prms->in_w);
  
  //Filter Descriptor
  cudnnCreateFilterDescriptor(&filt_desc);
  cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, prms->filter_n, prms->filter_c, prms->filter_h, prms->filter_w);
 
  //Convolution
  cudnnCreateConvolutionDescriptor(&conv_desc);
  cudnnSetConvolution2dDescriptor(conv_desc, prms->pad_h, prms->pad_w, prms->u, prms->v, prms->dil_h, prms->dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
 
  //Out
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->in_n, prms->in_c, prms->out_h,  prms->out_w);

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
  }
  else 
  {
    cudnnConvolutionForward(cudnn[id], &prms->alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, 
                           fwd_algo, NULL, 0, &prms->beta, out_desc, out_data);
  }

  //Bias Descriptor
  cudnnCreateTensorDescriptor(&bias_desc);
  cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

  cudnnAddTensor(cudnn[id], &prms->alpha, bias_desc, bias_data, &prms->alpha, out_desc, out_data);

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyFilterDescriptor(filt_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
}

tensor submit_convolution2D_forward(const int pad_h, const int pad_w, const int u, const int v, const int dil_h, const int dil_w, 
const float alpha, const float beta, const tensor in, const tensor filter, const tensor bias)
{
  tensor out = (tensor)malloc(sizeof(tensor));

  out->n = in->n;
  out->c = in->c;
  out->h = 1 + ( in->h + 2*pad_h - (((filter->h-1)*dil_h)+1) )/u;
  out->w = 1 + ( in->w + 2*pad_w - (((filter->w-1)*dil_w)+1) )/v;

  const struct convolution2D_forward_params prms = {alpha, beta, 
                                            in->n, in->c, in->h, in->w, 
                                            filter->n, filter->c, filter->h, filter->w,
                                            pad_h, pad_w, u, v, dil_h, dil_w,
                                            out->n, out->c, out->h, out->w};

  //Tensor out
  starpu_vector_data_register(&out->handle, -1, NULL, out->n * out->c * out->h * out->w, sizeof(float));

  struct starpu_task *task = starpu_task_create();
  task->cl = &convolution2D_forward_cl;
  task->handles[0] = in->handle;
  task->handles[1] = filter->handle;
  task->handles[2] = bias->handle;
  task->handles[3] = out->handle;
  task->cl_arg = &prms;
  task->cl_arg_size = sizeof(const struct convolution2D_forward_params);
  task->destroy = 0;

  const int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  starpu_task_wait_for_all();

  return out;
}


//------- POOLING --------
void pooling2D_forward(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  float *bias_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[1]); 
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[2]); 
  const struct pooling2D_forward_params *prms = (struct pooling2D_forward_params *)_args;
  const int id = starpu_worker_get_id();

  cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
  cudnnPoolingDescriptor_t pool_desc;

  //In Descriptor
  cudnnCreateTensorDescriptor(&in_desc);
  cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->in_n, prms->in_c, prms->in_h, prms->in_w);
 
  //Max Pooling
  cudnnCreatePoolingDescriptor(&pool_desc);
  cudnnSetPooling2dDescriptor(pool_desc, prms->mode, prms->maxpoolingNanOpt, prms->windowHeight, prms->windowWidth, prms->verticalPadding, prms->horizontalPadding, prms->verticalStride, prms->horizontalStride);

  //Out Descriptor
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->out_n, prms->out_c, prms->out_h, prms->out_w);
 
  cudnnPoolingForward(cudnn[id], pool_desc, &prms->alpha, in_desc, in_data, &prms->beta, out_desc, out_data);

  //Bias Descriptor
  cudnnCreateTensorDescriptor(&bias_desc);
  cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

  cudnnAddTensor(cudnn[id], &prms->alpha, bias_desc, bias_data, &prms->alpha, out_desc, out_data);

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pool_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
}

tensor submit_max_pooling2D_forward(const int windowHeight, const int windowWidth, const int verticalPadding, const int horizontalPadding, 
const int verticalStride, const int horizontalStride, const float alpha, const float beta, const tensor in, const tensor bias) 
{
  tensor out = (tensor)malloc(sizeof(tensor));
  out->n = in->n;
  out->c = in->c;
  out->h = 1 + (in->h + 2*horizontalPadding - windowHeight)/horizontalStride;
  out->w = 1 + (in->w + 2*verticalPadding - windowWidth)/verticalStride;

  const struct pooling2D_forward_params prms = {alpha, beta,
                                        in->n, in->c, in->h, in->w,
                                        CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                        windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride,
                                        out->n, out->c, out->h, out->w};

  //Tensor out
  starpu_vector_data_register(&out->handle, -1, NULL, out->n * out->c * out->h * out->w, sizeof(float));

  struct starpu_task *task = starpu_task_create();
  task->cl = &pooling2D_forward_cl;
  task->handles[0] = in->handle;
  task->handles[1] = bias->handle;
  task->handles[2] = out->handle;
  task->cl_arg = &prms;
  task->cl_arg_size = sizeof(struct pooling2D_forward_params);
  task->destroy = 0;

  const int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
  
  starpu_task_wait_for_all();

  return out;
}
