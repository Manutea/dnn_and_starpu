#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cudnn.h"

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define REQUESTED_ALGO 10

struct convolution_params
{
  int in_n, in_c, in_h, in_w;                     // input
  int filt_k, filt_c, filt_h, filt_w;             // filter
  int pad_h, pad_w, str_h, str_w, dil_h, dil_w;   // convolution
  int out_n, out_c, out_h, out_w;                 // out
  size_t ws_size, in_size, filt_size, out_size;

  float alpha, beta;    //Pointers to scaling factors

  cudnnHandle_t cudnn[STARPU_NMAXWORKERS];

  cudnnTensorDescriptor_t in_desc;                
  cudnnFilterDescriptor_t filt_desc;        
  cudnnConvolutionDescriptor_t conv_desc;       
  cudnnTensorDescriptor_t out_desc;
};

void init_cudnn(void *);
void conv(void **, void *);
void free_conv(const struct convolution_params *);
float *out_to_in(starpu_data_handle_t *, struct convolution_params *);
void submit_conv(const starpu_data_handle_t, const starpu_data_handle_t, const starpu_data_handle_t, struct convolution_params *);
float *init_conv(float *, starpu_data_handle_t *, float *, starpu_data_handle_t *, starpu_data_handle_t *, struct convolution_params *);

static struct starpu_perfmodel conv_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "conv_model"
};

static struct starpu_codelet conv_cl =
{
  .cuda_funcs = {conv},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 3,
  .modes = {STARPU_R, STARPU_R, STARPU_W},
  .model = &conv_model,
};

int main(void)
{
  struct convolution_params conv_params = {1, 1, 5, 5,             // input
                                                 1, 1, 2, 2,       // filter
                                                 1, 1, 1, 1, 1, 1, // convolution
                                                 0, 0, 0, 0,       // out
                                                 0, 25, 4, 0,      // workspace/in/filter/out size
                                                 1.0, 0.0};        //scaling factors        
                                        
  const int in_sz = conv_params.in_n * conv_params.in_c * conv_params.in_h * conv_params.in_w;
  float in_data[in_sz];
  for(int i=0; i<in_sz; i++) {
    in_data[i] = i;
  }

  const int flt_sz = conv_params.filt_k * conv_params.filt_c * conv_params.filt_h * conv_params.filt_w;
  float filt_data[flt_sz];
  for(int i=0; i<flt_sz; i++) {
    filt_data[i] = 1.0f;
  }

  int ret = starpu_init(NULL);
  if (ret == -ENODEV)
  {
    return 77;
  }

  int gpuprocs[STARPU_NMAXWORKERS];
  unsigned ngpus =  starpu_cuda_worker_get_count();
  starpu_worker_get_ids_by_type(STARPU_CUDA_WORKER, gpuprocs, ngpus);

  starpu_execute_on_each_worker(init_cudnn, conv_params.cudnn, STARPU_CUDA);

  starpu_data_handle_t in_data_handle;
  starpu_data_handle_t filt_data_handle;
  starpu_data_handle_t out_data_handle;

  float *out_data = init_conv(in_data, &in_data_handle, filt_data, &filt_data_handle, &out_data_handle, &conv_params);
  submit_conv(in_data_handle, filt_data_handle, out_data_handle, &conv_params);

  //out_to_in(&out_data_handle, &conv_params);
  //submit_conv(in_data_handle, filt_data_handle, out_data_handle, &conv_params);

  starpu_data_unregister(in_data_handle);
  starpu_data_unregister(filt_data_handle);
  starpu_data_unregister(out_data_handle);

  starpu_memory_unpin(in_data, sizeof(in_data[0])*conv_params.in_size);
  starpu_memory_unpin(filt_data, sizeof(filt_data[0])*conv_params.filt_size);
  starpu_memory_unpin(out_data, sizeof(out_data[0])*conv_params.out_size);

  free_conv(&conv_params);

  starpu_shutdown();

  // --------------- RESULT -------------
  printf("\n");
  printf("out_n : %d\n", conv_params.out_n);
  printf("out_c : %d\n", conv_params.out_c);
  printf("out_h : %d\n", conv_params.out_h);
  printf("out_w : %d\n", conv_params.out_w);
  printf("\n");

  for(int i=0; i<conv_params.out_size; i++) 
  {
    printf("%f \n", out_data[i]);
  }

  free(out_data);
  return 0;
}

void init_cudnn(void *arg) 
{
  cudnnHandle_t *cudnn = (cudnnHandle_t *) arg;
  const int id = starpu_worker_get_id();
  cudnnCreate(cudnn[id]);
  cudnnSetStream(cudnn[id], starpu_cuda_get_local_stream());
}

float *out_to_in(starpu_data_handle_t * out_h, struct convolution_params *prms)
{
  prms->in_size = prms->out_size;
  prms->in_n = prms->out_n;
  prms->in_c = prms->out_c;
  prms->in_h = prms->out_h;
  prms->in_w = prms->out_w;
  prms->in_desc = &prms->out_desc;

  cudnnGetConvolution2dForwardOutputDim(prms->conv_desc, prms->in_desc, prms->filt_desc, &prms->out_n, &prms->out_c, &prms->out_h, &prms->out_w);
  prms->out_size = prms->out_n * prms->out_c * prms->out_h * prms->out_w;
  float *out = (float *) malloc(prms->out_size * sizeof(float));

  cudnnSetTensor4dDescriptor(prms->out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->out_n, prms->out_c, prms->out_h, prms->out_w);

  starpu_memory_pin(out, sizeof(out[0]) * prms->out_size);
  starpu_vector_data_register(out_h, STARPU_MAIN_RAM, (uintptr_t)out, prms->out_size, sizeof(out[0]));

  return out;
}

float *init_conv(float *in, starpu_data_handle_t *in_h, float *filt, starpu_data_handle_t *filt_h, starpu_data_handle_t *out_h, struct convolution_params *prms)
{
  prms->iBufferIn = 0;

  //Tensor in
  cudnnCreateTensorDescriptor(&prms->in_desc);
  cudnnSetTensor4dDescriptor(prms->in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->in_n, prms->in_c, prms->in_h, prms->in_w);

  //Filter
  cudnnCreateFilterDescriptor(&prms->filt_desc);
  cudnnSetFilter4dDescriptor(prms->filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, prms->filt_k, prms->filt_c, prms->filt_h, prms->filt_w);

  //Convolution
  cudnnCreateConvolutionDescriptor(&prms->conv_desc);
  cudnnSetConvolution2dDescriptor(prms->conv_desc, prms->pad_h, prms->pad_w, prms->str_h, prms->str_w,
  prms->dil_h, prms->dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

  //Setup the output tensor and allocate the proper amount of memory prior to launch the actual convolution
  cudnnGetConvolution2dForwardOutputDim(prms->conv_desc, prms->in_desc, prms->filt_desc, &prms->out_n, &prms->out_c, &prms->out_h, &prms->out_w);
  prms->out_size = prms->out_n * prms->out_c * prms->out_h * prms->out_w;
  float *out = (float *) malloc(prms->out_size * sizeof(float));

  //Tensor out
  cudnnCreateTensorDescriptor(&prms->out_desc);
  cudnnSetTensor4dDescriptor(prms->out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->out_n, prms->out_c, prms->out_h, prms->out_w);
  
  starpu_memory_pin(in, sizeof(in[0]) * prms->in_size);
  starpu_vector_data_register(in_h, STARPU_MAIN_RAM, (uintptr_t)in, prms->in_size, sizeof(in[0]));

  starpu_memory_pin(filt, sizeof(filt[0]) * prms->filt_size);
  starpu_vector_data_register(filt_h, STARPU_MAIN_RAM, (uintptr_t)filt, prms->filt_size, sizeof(filt[0]));

  starpu_memory_pin(out, sizeof(out[0]) * prms->out_size);
  starpu_vector_data_register(out_h, STARPU_MAIN_RAM, (uintptr_t)out, prms->out_size, sizeof(out[0]));

  return out;
}

void conv(void *buffers[], void *_args)
{
  const float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  const float *filt_data  = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[2]); 
  struct convolution_params *prms = (struct convolution_params *)_args;

  const int id = starpu_worker_get_id();

  cudnnCreate(&prms->cudnn[id]);
  cudnnSetStream(prms->cudnn[id], starpu_cuda_get_local_stream());

  //This function attempts all algorithms available for cudnnConvolutionForward().
  int n_returnedAlgo;  
  cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf[REQUESTED_ALGO];
  cudnnFindConvolutionForwardAlgorithm(prms->cudnn[id], prms->in_desc, prms->filt_desc, prms->conv_desc, prms->out_desc, REQUESTED_ALGO, &n_returnedAlgo, fwd_algo_perf);

  //This function returns the amount of GPU memory workspace the user needs to allocate to be able to call cudnnConvolutionForward() with the specified algorithm.
  cudnnConvolutionFwdAlgo_t fwd_algo = fwd_algo_perf[0].algo;
  cudnnGetConvolutionForwardWorkspaceSize(prms->cudnn[id], prms->in_desc, prms->filt_desc, prms->conv_desc, prms->out_desc, fwd_algo, &prms->ws_size);

  printf("\nCudnn predicted Time : %f \n", fwd_algo_perf[0].time);

  if(prms->ws_size > 0)
  {
    float *ws_data;
    cudaMalloc(&ws_data, prms->ws_size);
    cudnnConvolutionForward(prms->cudnn[id], &prms->alpha, prms->in_desc, in_data, prms->filt_desc, filt_data, prms->conv_desc, 
                            fwd_algo, ws_data, prms->ws_size, &prms->beta, prms->out_desc, out_data);
    cudaFree(ws_data);
  }
  else 
  {
    cudnnConvolutionForward(prms->cudnn[id], &prms->alpha, prms->in_desc, in_data, prms->filt_desc, filt_data, prms->conv_desc, 
                            fwd_algo, NULL, prms->ws_size, &prms->beta, prms->out_desc, out_data);
  }
}

void submit_conv(const starpu_data_handle_t in, const starpu_data_handle_t filt, const starpu_data_handle_t out, struct convolution_params *prms)
{
  struct starpu_task *task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &conv_cl;
  task->handles[0] = in;
  task->handles[1] = filt;
  task->handles[2] = out;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct convolution_params);

  int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

void free_conv(const struct convolution_params *prms) 
{
  cudnnDestroyTensorDescriptor(prms->out_desc);
  cudnnDestroyConvolutionDescriptor(prms->conv_desc);
  cudnnDestroyFilterDescriptor(prms->filt_desc);
  cudnnDestroyTensorDescriptor(prms->in_desc);

  for(int i = 0; i < starpu_cuda_worker_get_count(); i++)
  {
    cudnnDestroy(prms->cudnn[i]);
  }
}