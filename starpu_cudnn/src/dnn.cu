#include <starpu.h>

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