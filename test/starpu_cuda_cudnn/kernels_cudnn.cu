#include <starpu.h>
#include "cudnn.h"

struct cudnn_convolution_params
{
  // input
  int in_n, in_c, in_h, in_w;
  // filter
  int filt_k, filt_c, filt_h, filt_w;
  // convolution
  int pad_h, pad_w, str_h, str_w, dil_h, dil_w;
  // out
  int out_n, out_c, out_h, out_w;
  // workspace size
  size_t ws_size;
};

extern "C" void init_conv_cudnn_func(void *buffers[], void *_args)
{
  //Tensor In, Filter, Convolution params
  cudnn_convolution_params *prms = (cudnn_convolution_params *)_args;

  float *in_data    = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  float *filt_data  = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
  float *out_data   = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  //Tensor in
  cudnnTensorDescriptor_t in_desc;
  cudnnCreateTensorDescriptor(&in_desc);
  cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->in_n, prms->in_c, prms->in_h, prms->in_w);

  //Filter
  cudnnFilterDescriptor_t filt_desc;
  cudnnCreateFilterDescriptor(&filt_desc);
  cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, prms->filt_k, prms->filt_c, prms->filt_h, prms->filt_w);

  //Convoluion
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  cudnnSetConvolution2dDescriptor(conv_desc, prms->pad_h, prms->pad_w, prms->str_h, prms->str_w,
  prms->dil_h, prms->dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

  //Setup the output tensor and allocate the proper amount of memory prior to launch the actual convolution
  int out_n, out_c, out_h, out_w;
  cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &prms->out_n, &prms->out_c, &prms->out_h, &prms->out_w);

  //Tensor out
  cudnnTensorDescriptor_t out_desc;
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, prms->out_n, prms->out_c, prms->out_h, prms->out_w);
  
  //This function attempts all algorithms available for cudnnConvolutionForward().
  int n_returnedAlgo;  
  const int n_requestedAlgo = 10;
  cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf[n_requestedAlgo];
  cudnnFindConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, n_requestedAlgo, &n_returnedAlgo, fwd_algo_perf);

  //This function returns the amount of GPU memory workspace the user needs to allocate to be able to call cudnnConvolutionForward() with the specified algorithm.
  cudnnConvolutionFwdAlgo_t fwd_algo = fwd_algo_perf[0].algo;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, fwd_algo, &prms->ws_size);

  //float *ws_data;
  //cudaMalloc(&ws_data, prms_in->ws_size);

  // perform
  float alpha = 1.f;
  float beta  = 0.f;

  cudnnConvolutionForward(cudnn, &alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, fwd_algo, NULL/*ws_data*/, prms->ws_size, &beta, out_desc, out_data);

  // results  
  //std::cout << "in_data:" << std::endl;
  //print(in_data, in_n, in_c, in_h, in_w);
  //
  //std::cout << "filt_data:" << std::endl;
  //print(filt_data, filt_k, filt_c, filt_h, filt_w);
  //
  //std::cout << "out_data:" << std::endl;
  //print(out_data, out_n, out_c, out_h, out_w);

  // finalizing
  //cudnnDestroyTensorDescriptor(out_desc);
  //cudnnDestroyConvolutionDescriptor(conv_desc);
  //cudnnDestroyFilterDescriptor(filt_desc);
  //cudnnDestroyTensorDescriptor(in_desc);
  //cudnnDestroy(cudnn);
}