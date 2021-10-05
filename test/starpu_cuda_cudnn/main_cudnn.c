#include <starpu.h>
#include <math.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void cuda_dev_const(void *descr[], void *_args);

typedef void (*device_func)(void **, void *);

int execute_on(uint32_t where, device_func func, float *filt_data, int pnx, int pny)
{
  struct starpu_codelet cl;
  starpu_data_handle_t filt_data_handle;
  int i;

  starpu_matrix_data_register(&filt_data_handle, STARPU_MAIN_RAM, (uintptr_t)filt_data, pnx, pnx, pny, sizeof(float));

  starpu_codelet_init(&cl);
  cl.where = where;
  cl.cuda_funcs[0] = func;
  cl.nbuffers = 1;
  cl.modes[0] = STARPU_RW,
  cl.model = NULL;
  cl.name = "dev_const";

  struct starpu_task *task = starpu_task_create();
  task->cl = &cl;
  task->callback_func = NULL;
  task->handles[0] = filt_data_handle;

  int ret = starpu_task_submit(task);
  if (STARPU_UNLIKELY(ret == -ENODEV))
  {
    printf("\n\n");
    FPRINTF(stderr, "No worker may execute this task\n");
    task->destroy = 0;
    starpu_task_destroy(task);
    return 1;
  }

  starpu_task_wait_for_all();

  /* update the array in RAM */
  starpu_data_unregister(filt_data_handle);

  return 0;
}

int main(void)
{
  const int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
  float *filt_data;
  int ret;
  int nx=filt_w * filt_h;
  int ny=filt_k * filt_c;

  //Init StarPu
  ret = starpu_init(NULL);
  if (ret == -ENODEV)
    return 77;
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");


  filt_data = (float*)malloc(nx*ny*sizeof(float));
  assert(filt_data);

  ret = execute_on(STARPU_CUDA, cuda_dev_const, filt_data, nx, ny);

  free(filt_data);
  starpu_shutdown();

  return 0;
}
