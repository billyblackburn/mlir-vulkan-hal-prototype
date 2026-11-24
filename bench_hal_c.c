#define _POSIX_C_SOURCE 200809L
#include "vk_hal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUP_ITERS 10
#define BENCH_ITERS 100

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

typedef struct {
  uint64_t ptr_a;
  uint64_t ptr_b;
  uint64_t ptr_c;
  float alpha;
  uint32_t n;
} SaxpyBdaArgs;

static void bench_saxpy_and_bda(VkHalContext *ctx) {
  const uint32_t N = 4u * 1024u * 1024u;
  const uint32_t WG = 256u;
  const uint32_t GRID_X = (N + WG - 1u) / WG;
  const uint64_t bytes = (uint64_t)N * sizeof(float);
  const double bytes_per_iter = 6.0 * (double)bytes; // 96 MiB = 100.66 MB

  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_out = (float *)malloc(bytes);
  for (uint32_t i = 0; i < N; i++) {
    float v = (float)(i & 1023u);
    h_a[i] = v;
    h_b[i] = v * 0.5f;
  }

  VkHalBuffer *d_a = vk_hal_buffer_create(ctx, bytes);
  VkHalBuffer *d_b = vk_hal_buffer_create(ctx, bytes);
  VkHalBuffer *d_c = vk_hal_buffer_create(ctx, bytes);
  VkHalBuffer *d_d = vk_hal_buffer_create(ctx, bytes);
  vk_hal_buffer_upload(d_a, h_a, bytes);
  vk_hal_buffer_upload(d_b, h_b, bytes);

  VkHalQueue *q1 = vk_hal_queue_create(ctx);
  VkHalQueue *q2 = vk_hal_queue_create(ctx);
  VkHalEvent *ev = vk_hal_event_create(ctx);

  // 1. Descriptor-set batched SAXPY: MLIR vs GLSL
  const char *labels[2] = {"SAXPY 4M MLIR GPU->SPIR-V", "SAXPY 4M GLSL (glslc -O) "};
  const char *paths[2] = {"bench_mlir.spv", "bench_glsl.spv"};
  const char *entries[2] = {"saxpy", "main"};
  VkHalBuffer *pass1[3] = {d_a, d_b, d_c};
  VkHalBuffer *pass2[3] = {d_c, d_b, d_d};

  for (int k = 0; k < 2; k++) {
    VkHalBundle *b = vk_hal_bundle_load(ctx, paths[k]);
    VkHalFunction *fn = vk_hal_function_load(b, entries[k], 3);
    for (int i = 0; i < WARMUP_ITERS; i++) {
      vk_hal_queue_execute(q1, fn, pass1, 3, GRID_X, 1, 1);
      vk_hal_queue_execute(q1, fn, pass2, 3, GRID_X, 1, 1);
      vk_hal_queue_synchronize(q1);
    }
    double t0 = now_ms();
    for (int i = 0; i < BENCH_ITERS; i++) {
      vk_hal_queue_execute(q1, fn, pass1, 3, GRID_X, 1, 1);
      vk_hal_queue_execute(q1, fn, pass2, 3, GRID_X, 1, 1);
      vk_hal_queue_synchronize(q1);
    }
    double avg_ms = (now_ms() - t0) / BENCH_ITERS;
    double gbps = (bytes_per_iter / 1e9) / (avg_ms / 1e3);

    vk_hal_buffer_download(d_d, h_out, bytes);
    float max_err = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
      float exp = 2.5f * (2.5f * h_a[i] + h_b[i]) + h_b[i];
      float diff = fabsf(h_out[i] - exp);
      if (diff > max_err)
        max_err = diff;
    }
    printf("[HAL Queue] %s: %6.3f ms/iter | %6.2f GB/s | max_err=%.6f\n",
           labels[k], avg_ms, gbps, max_err);
    vk_hal_function_unload(fn);
    vk_hal_bundle_unload(b);
  }

  // 2. BDA + Push Constants + Cross-Queue Timeline Semaphore Sync
  VkHalBundle *b_bda = vk_hal_bundle_load(ctx, "saxpy_bda.spv");
  VkHalFunction *fn_bda =
      vk_hal_function_load_bda(b_bda, "main", (uint32_t)sizeof(SaxpyBdaArgs));

  SaxpyBdaArgs a1 = {vk_hal_buffer_device_address(d_a),
                     vk_hal_buffer_device_address(d_b),
                     vk_hal_buffer_device_address(d_c), 2.5f, N};
  SaxpyBdaArgs a2 = {vk_hal_buffer_device_address(d_c),
                     vk_hal_buffer_device_address(d_b),
                     vk_hal_buffer_device_address(d_d), 3.0f, N};

  vk_hal_queue_execute_bda(q1, fn_bda, &a1, sizeof(a1), GRID_X, 1, 1);
  vk_hal_queue_record_event(q1, ev);
  vk_hal_queue_wait_event(q2, ev);
  vk_hal_queue_execute_bda(q2, fn_bda, &a2, sizeof(a2), GRID_X, 1, 1);
  vk_hal_queue_synchronize(q2);

  vk_hal_buffer_download(d_d, h_out, bytes);
  float max_err_bda = 0.0f;
  for (uint32_t i = 0; i < N; i++) {
    float exp = 3.0f * (2.5f * h_a[i] + h_b[i]) + h_b[i];
    float diff = fabsf(h_out[i] - exp);
    if (diff > max_err_bda)
      max_err_bda = diff;
  }

  for (int i = 0; i < WARMUP_ITERS; i++) {
    vk_hal_queue_execute_bda(q1, fn_bda, &a1, sizeof(a1), GRID_X, 1, 1);
    vk_hal_queue_execute_bda(q1, fn_bda, &a2, sizeof(a2), GRID_X, 1, 1);
    vk_hal_queue_synchronize(q1);
  }
  double t0 = now_ms();
  for (int i = 0; i < BENCH_ITERS; i++) {
    vk_hal_queue_execute_bda(q1, fn_bda, &a1, sizeof(a1), GRID_X, 1, 1);
    vk_hal_queue_execute_bda(q1, fn_bda, &a2, sizeof(a2), GRID_X, 1, 1);
    vk_hal_queue_synchronize(q1);
  }
  double avg_bda = (now_ms() - t0) / BENCH_ITERS;
  double gbps_bda = (bytes_per_iter / 1e9) / (avg_bda / 1e3);
  printf("[HAL BDA  ] SAXPY 4M (0 descriptors)   : %6.3f ms/iter | %6.2f GB/s | "
         "cross-q event=%d max_err=%.6f\n",
         avg_bda, gbps_bda, vk_hal_event_is_ready(ev), max_err_bda);

  vk_hal_function_unload(fn_bda);
  vk_hal_bundle_unload(b_bda);
  vk_hal_event_destroy(ev);
  vk_hal_queue_destroy(q2);
  vk_hal_queue_destroy(q1);
  vk_hal_buffer_destroy(d_d);
  vk_hal_buffer_destroy(d_c);
  vk_hal_buffer_destroy(d_b);
  vk_hal_buffer_destroy(d_a);
  free(h_a);
  free(h_b);
  free(h_out);
}

static void bench_matmul(VkHalContext *ctx) {
  const uint32_t M = 512u;
  const uint32_t ELEMS = M * M;
  const uint32_t GRID = M / 16u;
  const uint64_t bytes = (uint64_t)ELEMS * sizeof(float);
  const double flops = 2.0 * (double)M * (double)M * (double)M;

  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c = (float *)malloc(bytes);
  for (uint32_t r = 0; r < M; r++) {
    for (uint32_t c = 0; c < M; c++) {
      h_a[r * M + c] = (float)((int)((r + c) % 17u) - 8) * 0.125f;
      h_b[r * M + c] = (float)((int)((r * 3u + c) % 13u) - 6) * 0.125f;
    }
  }

  VkHalBuffer *d_a = vk_hal_buffer_create(ctx, bytes);
  VkHalBuffer *d_b = vk_hal_buffer_create(ctx, bytes);
  VkHalBuffer *d_c = vk_hal_buffer_create(ctx, bytes);
  vk_hal_buffer_upload(d_a, h_a, bytes);
  vk_hal_buffer_upload(d_b, h_b, bytes);

  VkHalQueue *q = vk_hal_queue_create(ctx);
  VkHalBuffer *bufs[3] = {d_a, d_b, d_c};

  const char *labels[2] = {"MatMul 512x512 MLIR LDS  ",
                           "MatMul 512x512 GLSL LDS  "};
  const char *paths[2] = {"matmul_mlir.spv", "matmul_glsl.spv"};
  const char *entries[2] = {"matmul_tiled", "main"};

  for (int k = 0; k < 2; k++) {
    VkHalBundle *b = vk_hal_bundle_load(ctx, paths[k]);
    VkHalFunction *fn = vk_hal_function_load(b, entries[k], 3);
    for (int i = 0; i < WARMUP_ITERS; i++) {
      vk_hal_queue_execute(q, fn, bufs, 3, GRID, GRID, 1);
      vk_hal_queue_synchronize(q);
    }
    double t0 = now_ms();
    for (int i = 0; i < BENCH_ITERS; i++) {
      vk_hal_queue_execute(q, fn, bufs, 3, GRID, GRID, 1);
      vk_hal_queue_synchronize(q);
    }
    double avg_ms = (now_ms() - t0) / BENCH_ITERS;
    double gflops = (flops / 1e9) / (avg_ms / 1e3);

    vk_hal_buffer_download(d_c, h_c, bytes);
    float max_err = 0.0f;
    for (uint32_t r = 0; r < 16u; r++) {
      for (uint32_t c = 0; c < M; c++) {
        float ref = 0.0f;
        for (uint32_t kk = 0; kk < M; kk++)
          ref += h_a[r * M + kk] * h_b[kk * M + c];
        float diff = fabsf(h_c[r * M + c] - ref);
        if (diff > max_err)
          max_err = diff;
      }
    }
    printf("[HAL Queue] %s: %6.3f ms/iter | %6.2f GFLOP/s | max_err=%.6f\n",
           labels[k], avg_ms, gflops, max_err);
    vk_hal_function_unload(fn);
    vk_hal_bundle_unload(b);
  }

  vk_hal_queue_destroy(q);
  vk_hal_buffer_destroy(d_c);
  vk_hal_buffer_destroy(d_b);
  vk_hal_buffer_destroy(d_a);
  free(h_a);
  free(h_b);
  free(h_c);
}

int main(void) {
  VkHalContext *ctx = vk_hal_context_create(0);
  if (!ctx) {
    fprintf(stderr, "Failed to create VkHalContext\n");
    return 1;
  }
  bench_saxpy_and_bda(ctx);
  bench_matmul(ctx);
  printf("[HAL Cache] Final VkPipelineCache size: %llu bytes\n",
         (unsigned long long)vk_hal_context_pipeline_cache_size(ctx));
  vk_hal_context_destroy(ctx);
  return 0;
}
