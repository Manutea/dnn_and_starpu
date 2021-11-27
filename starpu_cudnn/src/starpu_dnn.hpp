#ifndef STARPU_DNN_HPP
#define STARPU_DNN_HPP

#include <starpu.h>
#include <starpu_cublas_v2.h>

#include "cudnn.h"

#define REQUESTED_ALGO 10

cudnnHandle_t cudnn[STARPU_NMAXWORKERS];

struct _tensor 
{
  float *data;
  starpu_data_handle_t handle;
  int n, c, h, w;
};
typedef struct _tensor tensor;

int starpu_dnn_init();
void starpu_dnn_shutdown();
void cudnn_shutdown();
void cudnn_init(void *);
void free_tensor(tensor *);
tensor *init_tensor(float *, int, int, int, int);

// Convolution Forward
struct convolution2D_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
  int filter_n, filter_c, filter_h, filter_w;
  int pad_h, pad_w, u, v, dil_h, dil_w;
  int out_n, out_c, out_h, out_w;
};
void convolution2D_forward(void **, void *);
tensor *submit_convolution2D_forward(int, int, int, int, int, int, float, float, const tensor *, const tensor *, const tensor *);
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

// Max pooling Forward
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
tensor *submit_max_pooling2D_forward(int, int, int, int, int, int, float, float, const tensor *);
static struct starpu_perfmodel pooling2D_forward_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "pooling2D_forward_model"
};
static struct starpu_codelet pooling2D_forward_cl =
{
  .cuda_funcs = {pooling2D_forward},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 2,
  .modes = {STARPU_R, STARPU_W},
  .model = &pooling2D_forward_model,
};

// ReLU activation Forward
struct relu_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
};
void relu_forward(void **, void *);
tensor *submit_relu_forward(float, float, const tensor *);
static struct starpu_perfmodel relu_forward_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "relu_forward_model"
};
static struct starpu_codelet relu_forward_cl =
{
  .cuda_funcs = {relu_forward},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 2,
  .modes = {STARPU_R, STARPU_W},
  .model = &relu_forward_model,
};

// Softmax Forward
struct softmax_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
};
void softmax_forward(void **, void *);
tensor *submit_softmax_forward(float, float, const tensor *);
static struct starpu_perfmodel softmax_forward_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "softmax_forward_model"
};
static struct starpu_codelet softmax_forward_cl =
{
  .cuda_funcs = {softmax_forward},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 2,
  .modes = {STARPU_R, STARPU_W},
  .model = &softmax_forward_model,
};

// Fully connected Forward
struct fullyco_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
  int bias_h, bias_w;
};
void fullyco_forward(void **, void *);
tensor *submit_fullyco_forward(float alpha, float beta, const tensor *in, const tensor *bias);
static struct starpu_perfmodel fullyco_forward_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "fullyco_forward_model"
};
static struct starpu_codelet fullyco_forward_cl =
{
  .cuda_funcs = {fullyco_forward},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 2,
  .modes = {STARPU_R, STARPU_R, STARPU_W},
  .model = &fullyco_forward_model,
};


//------- Initialize --------
int starpu_dnn_init()
{
  const int ret = starpu_init(NULL);
  if (ret == -ENODEV)
  {
      return 77;
  }
  starpu_execute_on_each_worker(cudnn_init, cudnn, STARPU_CUDA);
  starpu_cublas_init();

  return 0;
}

void starpu_dnn_shutdown()
{
  cudnn_shutdown();
  starpu_cublas_shutdown();
  starpu_shutdown();
}

void cudnn_init(void *arg) 
{
  cudnnHandle_t *cudnn_ = (cudnnHandle_t *) arg;
  const int id = starpu_worker_get_id();
  cudnnCreate(&cudnn_[id]);
  cudnnSetStream(cudnn_[id], starpu_cuda_get_local_stream());
}

void free_tensor(tensor *t)
{
  if(t->data==nullptr)
  {
    starpu_data_unregister_submit(t->handle);
    free(t);
  }
  else
  {
    starpu_data_unregister(t->handle);
    starpu_free(t->data);
    free(t);
  }
  
}

void cudnn_shutdown() 
{
  for(int i = 0; i < starpu_cuda_worker_get_count(); i++)
  {
    cudnnDestroy(cudnn[i]);
  }
}

tensor *init_tensor(float *data, int n, int c, int h, int w)
{
  tensor *out = (tensor *)malloc(sizeof(tensor));
  out->n = n;
  out->c = c;
  out->h = h;
  out->w = w;
  out->data = data;

  if(data != NULL)
    starpu_vector_data_register(&out->handle, STARPU_MAIN_RAM, (uintptr_t)data, n * c * h * w, sizeof(float));
  else
    starpu_vector_data_register(&out->handle, -1, (uintptr_t) NULL, out->n * out->c * out->h * out->w, sizeof(float));

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
  size_t ws_size;
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

tensor *submit_convolution2D_forward(int pad_h, int pad_w, int u, int v, int dil_h, int dil_w, 
float alpha, float beta, const tensor *in, const tensor *filter, const tensor *bias)
{
  const int h = 1 + ( in->h + 2*pad_h - (((filter->h-1)*dil_h)+1) )/u;
  const int w = 1 + ( in->w + 2*pad_w - (((filter->w-1)*dil_w)+1) )/v;
  tensor *out = init_tensor(nullptr, in->n, in->c, h, w);

  struct convolution2D_forward_params *prms = (struct convolution2D_forward_params *)malloc(sizeof(struct convolution2D_forward_params));
  prms->alpha = alpha;
  prms->beta = beta;
  prms->in_n = in->n;
  prms->in_c = in->c;
  prms->in_h = in->h;
  prms->in_w = in->w;
  prms->filter_n = filter->n;
  prms->filter_c = filter->c;
  prms->filter_h = filter->h;
  prms->filter_w = filter->w;
  prms->pad_h = pad_h;
  prms->pad_w = pad_w;
  prms->u = u;
  prms->v = v;
  prms->dil_h = dil_h;
  prms->dil_w = dil_w;
  prms->out_n = out->n;
  prms->out_c = out->c;
  prms->out_h = out->h;
  prms->out_w = out->w;

  struct starpu_task *task = starpu_task_create();
  task->cl = &convolution2D_forward_cl;
  task->handles[0] = in->handle;
  task->handles[1] = filter->handle;
  task->handles[2] = bias->handle;
  task->handles[3] = out->handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(const struct convolution2D_forward_params);
  task->cl_arg_free = 1;

  const int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  return out;
}

//------- POOLING --------
void pooling2D_forward(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[1]); 
  const struct pooling2D_forward_params *prms = (struct pooling2D_forward_params *)_args;
  const int id = starpu_worker_get_id();

  cudnnTensorDescriptor_t in_desc, out_desc;
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

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pool_desc);
}

tensor *submit_max_pooling2D_forward(int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, 
 int verticalStride, int horizontalStride, float alpha, float beta, const tensor *in) 
{
  const int h = 1 + (in->h + 2*horizontalPadding - windowHeight)/horizontalStride;
  const int w = 1 + (in->w + 2*verticalPadding - windowWidth)/verticalStride;
  tensor *out = init_tensor(nullptr, in->n, in->c, h, w);

  struct pooling2D_forward_params *prms = (struct pooling2D_forward_params *)malloc(sizeof(struct pooling2D_forward_params));
  prms->alpha = alpha;
  prms->beta = beta;
  prms->in_n = in->n;
  prms->in_c = in->c;
  prms->in_h = in->h;
  prms->in_w = in->w;
  prms->mode = CUDNN_POOLING_MAX;
  prms->maxpoolingNanOpt = CUDNN_NOT_PROPAGATE_NAN;
  prms->windowHeight = windowHeight;
  prms->windowWidth = windowWidth;
  prms->verticalPadding = verticalPadding;
  prms->horizontalPadding = horizontalPadding;
  prms->verticalStride = verticalStride;
  prms->horizontalStride = horizontalStride;
  prms->out_n = out->n;
  prms->out_c = out->c;
  prms->out_h = out->h;
  prms->out_w = out->w;

  struct starpu_task *task = starpu_task_create();
  task->cl = &pooling2D_forward_cl;
  task->handles[0] = in->handle;
  task->handles[1] = out->handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct pooling2D_forward_params);
  task->cl_arg_free = 1;

  const int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  return out;
}

//------- RELU --------
void relu_forward(void *buffers[], void *_args) 
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[1]); 
  const struct relu_forward_params *prms = (struct relu_forward_params *)_args;
  const int id = starpu_worker_get_id();
  
  cudnnActivationDescriptor_t activation_desc;
  cudnnCreateActivationDescriptor(&activation_desc);
  cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

  cudnnTensorDescriptor_t tensor_desc;
  cudnnCreateTensorDescriptor(&tensor_desc);
  cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->in_n, prms->in_c, prms->in_h, prms->in_w);

  cudnnActivationForward(cudnn[id], activation_desc, &prms->alpha, tensor_desc, in_data, &prms->beta, tensor_desc, out_data);

  cudnnDestroyActivationDescriptor(activation_desc);
  cudnnDestroyTensorDescriptor(tensor_desc);
}

tensor *submit_relu_forward(float alpha, float beta, const tensor *in) 
{
  tensor *out = init_tensor(NULL, in->n, in->c, in->h, in->w);

  struct relu_forward_params *prms = (struct relu_forward_params *)malloc(sizeof(struct relu_forward_params));
  prms->alpha = alpha;
  prms->alpha = beta;
  prms->in_n = in->n;
  prms->in_c = in->c;
  prms->in_h = in->h;
  prms->in_w = in->w;

  struct starpu_task *task = starpu_task_create();
  task->cl = &relu_forward_cl;
  task->handles[0] = in->handle;
  task->handles[1] = out->handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct relu_forward_params);
  task->cl_arg_free = 1;

  const int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  return out;
}

//------- SOFTMAX --------
void softmax_forward(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[1]); 
  const struct softmax_forward_params *prms = (struct softmax_forward_params *)_args;
  const int id = starpu_worker_get_id();

  cudnnTensorDescriptor_t tensor_desc;
  cudnnCreateTensorDescriptor(&tensor_desc);
  cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->in_n, prms->in_c, prms->in_h, prms->in_w);

  cudnnSoftmaxForward(cudnn[id], CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &prms->alpha, tensor_desc, in_data, &prms->beta, tensor_desc, out_data);

  cudnnDestroyTensorDescriptor(tensor_desc);
}

tensor *submit_softmax_forward(float alpha, float beta, const tensor *in) 
{
  tensor *out = init_tensor(nullptr, in->n, in->c, in->h, in->w);

  struct softmax_forward_params *prms = (struct softmax_forward_params *)malloc(sizeof(struct softmax_forward_params));
  prms->alpha = alpha;
  prms->beta = beta;
  prms->in_n = in->n;
  prms->in_c = in->c;
  prms->in_h = in->h;
  prms->in_w = in->w;

  struct starpu_task *task = starpu_task_create();
  task->cl = &softmax_forward_cl;
  task->handles[0] = in->handle;
  task->handles[1] = out->handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct softmax_forward_params);
  task->cl_arg_free = 1;

  const int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  return out;
}

//------- FULLY CONNECTED --------
void fullyco_forward(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  const float *bias_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[2]); 
  const struct fullyco_forward_params *prms = (struct fullyco_forward_params *)_args;

  starpu_cublas_set_stream();
  cublasHandle_t cublasHandle = starpu_cublas_get_local_handle();

  const int in_rows = prms->in_h * prms->in_w;
  const int lead_dim_bias = (prms->bias_h > prms->bias_w) ? prms->bias_h : prms->bias_w;

  cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            in_rows,                            //nb row of In and Out
            prms->bias_w,                       //nb col of Bias and Out
            in_rows,                            //nb col of In and row of Out
            &prms->alpha,                          
            in_data,                            //In
            in_rows,                            //Leading dimension of two-dimensional array used to store the matrix In
            bias_data,                          //Bias
            lead_dim_bias,                      //Leading dimension of two-dimensional array used to store the matrix Bias
            &prms->beta,
            out_data,                           //Out
            prms->bias_w);                      //Leading dimension of a two-dimensional array used to store the matrix C
}

tensor *submit_fullyco_forward(float alpha, float beta, const tensor *in, const tensor *bias)
{
  tensor *out = init_tensor(nullptr, in->n, in->c, 1, bias->w);

  struct fullyco_forward_params *prms = (struct fullyco_forward_params *)malloc(sizeof(struct fullyco_forward_params));
  prms->alpha = alpha;
  prms->beta = beta;
  prms->in_n = in->n;
  prms->in_c = in->c;
  prms->in_h = in->h;
  prms->in_w = in->w;
  prms->bias_h = bias->h;
  prms->bias_w = bias->w;

  struct starpu_task *task = starpu_task_create();
  task->cl = &softmax_forward_cl;
  task->handles[0] = in->handle;
  task->handles[1] = bias->handle;
  task->handles[2] = out->handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct fullyco_forward_params);
  task->cl_arg_free = 1;

  const int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  return out;
}

#endif
