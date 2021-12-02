#include "../include/starpu_dnn/starpu_dnn.hpp"
#include "cudnn.h"
#include "starpu_cublas_v2.h"

#define REQUESTED_ALGO 10
cudnnHandle_t cudnn[STARPU_NMAXWORKERS];


//------- SCHEDULER --------
void cudnn_init(void *arg) 
{
  cudnnHandle_t *cudnn_ = (cudnnHandle_t *) arg;
  const int id = starpu_worker_get_id();
  cudnnCreate(&cudnn_[id]);
  cudnnSetStream(cudnn_[id], starpu_cuda_get_local_stream());
}

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
  for(int i = 0; i < starpu_cuda_worker_get_count(); i++)
  {
    cudnnDestroy(cudnn[i]);
  }

  starpu_cublas_shutdown();
  starpu_shutdown();
}


//------- TENSOR --------
tensor *init_tensor(float *data, int n, int c, int h, int w)
{
  tensor *out = (tensor *)malloc(sizeof(tensor));
  out->n = n;
  out->c = c;
  out->h = h;
  out->w = w;
  out->data = data;

  if(data != nullptr)
  {
    starpu_memory_pin(data, n * c * h * w * sizeof(data[0]));
    starpu_vector_data_register(&out->handle, STARPU_MAIN_RAM, (uintptr_t)data, n * c * h * w, sizeof(float));
  }
  else
  {
    starpu_vector_data_register(&out->handle, -1, (uintptr_t) NULL, out->n * out->c * out->h * out->w, sizeof(float));
  }

  return out;
}

void free_tensor(tensor *t)
{
  if(t->data != nullptr)
  {
    starpu_data_unregister(t->handle);
    starpu_memory_unpin(t->data, t->n * t->c * t->h * t->w * sizeof(float));
    free(t);
  }
  else
  {
    starpu_data_unregister_submit(t->handle);   
    free(t);
  }
  
}


//------- CONVOLUTION --------
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

struct convolution2D_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
  int filter_n, filter_c, filter_h, filter_w;
  int pad_h, pad_w, u, v, dil_h, dil_w;
  int out_n, out_c, out_h, out_w;
};

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

struct pooling2D_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
  cudnnPoolingMode_t mode;
  cudnnNanPropagation_t maxpoolingNanOpt;
  int windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride;
  int out_n, out_c, out_h, out_w;
};

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

struct relu_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
};

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

struct softmax_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
};

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

struct fullyco_forward_params
{
  float alpha, beta;
  int in_n, in_c, in_h, in_w;
  int bias_h, bias_w;
};

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


//------- Linear -------- 
static struct starpu_perfmodel linear_forward_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "linear_forward_model"
};

static struct starpu_codelet linear_forward_cl = 
{
  .cuda_funcs = {linear_forward},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 5,
  .modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_W},
  .model = &linear_forward_model,
};

struct linear_forward_params
{                                                                                                                                                                                
  float alpha_w, beta_w, alpha_b, beta_b;  
  int in_n, in_c, in_h, in_w;
  int weight_n, weight_c, weight_h, weight_w;
  int bias_h, bias_w;
};

void linear_forward(void* buffers[], void* _args)
{
  const float *in_data     = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  const float *weight_data = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
  const float *bias_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);
  const float *one_vec_data = (float *)STARPU_VECTOR_GET_PTR(buffers[3]); 
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[4]);
  const struct linear_forward_params *prms = (struct linear_forward_params *)_args;

  starpu_cublas_set_stream(); 
  cublasHandle_t cublasHandle = starpu_cublas_get_local_handle();

  const int input_size = prms->in_c * prms->in_h * prms->in_w;
  const int output_size = prms->weight_h;
  //output = weights^T * input (without biases)
  cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
	      output_size,                        //nb row Weight and Out
	      prms->in_n,                         //nb col In and Out
	      input_size,                         //nb col Weight and row of In
	      &prms->alpha_w,                     //Must be 1
	      weight_data,                        //Weight data
	      input_size,                         //Leading dimension of two-dimensional array used to store the matrix Weight
	      in_data,                            //In data
	      input_size,                         //Leading dimension of two-dimensional array used to store the matrix In
	      &prms->beta_w,                      //Must be 0
	      out_data,                           //Out data
	      output_size);                       //Leading dimension of a two-dimensional array used to store the matrix Out

  // output += biases * d_one_vec^T
  cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
	      output_size,                        //nb row Bias and Out 
              prms->in_n,                         //nb col One_vec and Out 
	      1,                                  //nb col Weight and row of One_vec
              &prms->alpha_b,                     //Must be 1
	      bias_data,                          //Bias data
              output_size,                        //Leading dimension of two-dimensional array used to store the matrix Bias
              one_vec_data,                       //One_vec data
	      1,                                  //Leading dimension of two-dimensional array used to store the matrix One_vec
              &prms->beta_b,                      //Must be 1
              out_data,                           //Out data
              output_size);                       //Leading dimension of a two-dimensional array used to store the matrix Out
}

tensor *submit_linear_forward(const tensor *in, const tensor *weight, const tensor *bias, float alpha_w, float beta_w, float alpha_b, float beta_b)
{
  tensor *out = init_tensor(nullptr, in->n, in->c, bias->h, bias->w);

  //Move this in the init
  float *one_vec_data;
  starpu_malloc((void **)&one_vec_data, in->n*sizeof(float));
  for(int i=0; i<in->n; i++)
    one_vec_data[i] = 1.0f;
  tensor *one_vec = init_tensor(one_vec_data, in->n, 1, 1, 1);

  struct linear_forward_params *prms = (struct linear_forward_params *)malloc(sizeof(struct linear_forward_params));
  prms->alpha_w = alpha_w;
  prms->beta_w = beta_w;
  prms->alpha_b = alpha_b;
  prms->beta_b = beta_b;
  prms->in_n = in->n;
  prms->in_c = in->c;
  prms->in_h = in->h;
  prms->in_w = in->w;
  prms->weight_n = weight->n;
  prms->weight_c = weight->c;
  prms->weight_h = weight->h;
  prms->weight_w = weight->w;
  prms->bias_h = bias->h;
  prms->bias_w = bias->w;

  struct starpu_task *task = starpu_task_create();
  task->cl = &linear_forward_cl;
  task->handles[0] = in->handle;
  task->handles[1] = weight->handle;
  task->handles[2] = bias->handle;
  task->handles[3] = one_vec->handle;
  task->handles[4] = out->handle;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct linear_forward_params);
  task->cl_arg_free = 1;

  const int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  return out;
}