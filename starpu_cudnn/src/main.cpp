#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cudnn.h"

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define REQUESTED_ALGO 10

cudnnHandle_t cudnn[STARPU_NMAXWORKERS];
cudnnTensorDescriptor_t in_desc, out_desc;
cudnnFilterDescriptor_t filt_desc; 
cudnnConvolutionDescriptor_t conv_desc;

struct convolution_params
{
  size_t out_size;
  float alpha, beta;    //Pointers to scaling factors     
};

void init_cudnn(void *);
void conv(void **, void *);
void free_conv(const struct convolution_params *);
starpu_data_handle_t init_filter(float *, int, int, int, int, struct convolution_params *);
float *submit_conv(float *, int, int, int, int, int, int, int, int, int, int, starpu_data_handle_t, struct convolution_params *);

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
  int in_n = 1, in_c = 1, in_h = 5, in_w = 5;
  int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
  int pad_h = 1, pad_w = 1, str_h = 1, str_w = 1, dil_h = 1, dil_w = 1;
  struct convolution_params conv_params = {
                                                 0,      // workspace/in/filter/out size
                                                 1.0, 0.0};        //scaling factors        

  const int in_size = in_n * in_c * in_h * in_w;                    
  float in_data[in_size];
  for(int i=0; i<in_size; i++) {
    in_data[i] = i;
  }

  const int filt_size = filt_k * filt_c * filt_h * filt_w;  
  float filt_data[filt_size];
  for(int i=0; i<filt_size; i++) {
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
  starpu_execute_on_each_worker(init_cudnn, cudnn, STARPU_CUDA);

  starpu_data_handle_t filt_data_handle = init_filter(filt_data, filt_k, filt_c, filt_h, filt_w, &conv_params);
  float *out_data = submit_conv(in_data, in_n, in_c, in_h, in_w, pad_h, pad_w, str_h, str_w, dil_h, dil_w, filt_data_handle, &conv_params);

  starpu_data_unregister(filt_data_handle);
  starpu_memory_unpin(in_data, sizeof(in_data[0])*in_size);
  starpu_memory_unpin(filt_data, sizeof(filt_data[0])*filt_size);
  starpu_memory_unpin(out_data, sizeof(out_data[0])*conv_params.out_size);

  free_conv(&conv_params);

  starpu_shutdown();

  // --------------- RESULT -------------
  for(int i=0; i<conv_params.out_size; i++) 
  {
    printf("%f \n", out_data[i]);
  }

  free(out_data);
  return 0;
}

void init_cudnn(void *arg) 
{
  cudnnHandle_t *cudnn_ = (cudnnHandle_t *) arg;
  const int id = starpu_worker_get_id();
  cudnnCreate(&cudnn_[id]);
  cudnnSetStream(cudnn_[id], starpu_cuda_get_local_stream());
}

starpu_data_handle_t init_filter(float *filter, int k, int c, int h, int w, struct convolution_params *prms)
{
  starpu_data_handle_t filter_h;
  cudnnCreateFilterDescriptor(&filt_desc);
  cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w);
  const int filt_size = k * c * h * w;
  starpu_memory_pin(filter, sizeof(filter[0]) * filt_size);
  starpu_vector_data_register(&filter_h, STARPU_MAIN_RAM, (uintptr_t)filter, filt_size, sizeof(filter[0]));
  return filter_h;
}

void conv(void *buffers[], void *_args)
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
  int ws_size;
  cudnnConvolutionFwdAlgo_t fwd_algo = fwd_algo_perf[0].algo;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn[id], in_desc, filt_desc, conv_desc, out_desc, fwd_algo, &ws_size);

  printf("\nCudnn predicted Time : %f \n", fwd_algo_perf[0].time);

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
                            fwd_algo, NULL, ws_size, &prms->beta, out_desc, out_data);
  }
}

float * submit_conv(float *in, int in_n, int in_c, int in_h, int in_w, int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w, starpu_data_handle_t filt_hand, struct convolution_params *prms)
{
  starpu_data_handle_t in_hand, out_hand;

  //Convolution
  cudnnCreateConvolutionDescriptor(&conv_desc);
  cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

  //Tensor in
  cudnnCreateTensorDescriptor(&in_desc);
  cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w);
  const int in_size = in_n * in_c * in_h * in_w;
  starpu_memory_pin(in, sizeof(in[0]) * in_size);
  starpu_vector_data_register(&in_hand, STARPU_MAIN_RAM, (uintptr_t)in, in_size, sizeof(in[0]));

  //Setup the output tensor and allocate the proper amount of memory prior to launch the actual convolution
  int out_n, out_c, out_h, out_w;
  cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w);
  const int out_size = out_n * out_c * out_h * out_w;
  prms->out_size = out_size;
  float *out = (float *) malloc(out_size * sizeof(float));

  //Tensor out
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);
  starpu_memory_pin(out, sizeof(out[0]) * out_size);
  starpu_vector_data_register(&out_hand, STARPU_MAIN_RAM, (uintptr_t)out, out_size, sizeof(out[0]));

  struct starpu_task *task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &conv_cl;
  task->handles[0] = in_hand;
  task->handles[1] = filt_hand;
  task->handles[2] = out_hand;
  task->cl_arg = prms;
  task->cl_arg_size = sizeof(struct convolution_params);

  int ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  starpu_data_unregister(in_hand);
  starpu_data_unregister(out_hand);

  return out;
}

void free_conv(const struct convolution_params *prms) 
{
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyFilterDescriptor(filt_desc);
  cudnnDestroyTensorDescriptor(in_desc);

  for(int i = 0; i < starpu_cuda_worker_get_count(); i++)
  {
    cudnnDestroy(cudnn[i]);
  }
}
