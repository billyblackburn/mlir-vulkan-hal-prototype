# MLIR → SPIR-V → Vulkan Compute HAL Prototype

Reproducible prototype & benchmark exploring a Vulkan 1.3 compute backend aligned with Modular's [`SDK/include/HAL`](https://github.com/modular/modular/tree/main/SDK/include/HAL) (`modular/modular#7413`).

## What It Tests

1. **Direct MLIR `gpu` → `spirv` Lowering** (no GLSL intermediate):
   - 1D memory-bound 2-pass chained SAXPY (`N = 4,194,304` `f32`, `workgroup_size = 256`, 96 MiB / 100.66 MB per iter)
   - 2D compute + Workgroup LDS Tiled MatMul (`512×512`, `16×16` tiles, `memref.alloc() : memref<16x16xf32, #spirv.storage_class<Workgroup>>` + `gpu.barrier`)
2. **Persistent Vulkan 1.3 HAL (`libvk_hal.so`)**:
   - `Context` (`VkInstance`, `VkDevice` with Vulkan 1.2 `bufferDeviceAddress` + `timelineSemaphore`, `VkPipelineCache`)
   - `Queue` (batched async command stream with 4-slot command buffer ring, automatic `vkCmdPipelineBarrier`, and descriptor ring)
   - `Event` (timeline semaphore `record_event`, GPU-side cross-queue `wait_event`, `synchronize`, `is_ready`)
   - `Buffer` (`DEVICE_LOCAL` storage buffers + 64-bit GPU device addresses + staging host↔device transfers)
   - `Bundle` (`VkShaderModule`) & `Function` (descriptor-set or descriptorless BDA + push-constant pipelines)
3. **Head-to-head comparison** against `glslc -O` and pure-C Vulkan.

## Local Quick Start (Any Linux Vulkan 1.2+ Machine)

```bash
git clone <this-repo-url>
cd mlir_vulkan_compute
chmod +x run_all.sh
./run_all.sh
```

`run_all.sh` automatically detects system tools (`mlir-opt`, `mlir-translate`, `gcc`/`clang`, `glslc`/`glslangValidator`, `spirv-val`) or falls back to [Pixi](https://pixi.sh) (`pixi.toml` installs `mlir`, `c-compiler`, and `vulkan-headers` locally without touching system packages).

- `./bench_hal_c` runs the full HAL suite (SAXPY, 512×512 Tiled MatMul, BDA + push constants, and cross-queue timeline semaphores) with zero Mojo version dependency.
- `run_mojo.mojo`, `bench_mojo.mojo`, and `bench_bda_timeline.mojo` drive the exact same `libvk_hal.so` from Mojo via `sys.ffi.DLHandle`.

## Files

| File | Role |
| :--- | :--- |
| `vk_hal.h` / `vk_hal.c` | Persistent Vulkan 1.3 HAL (`Context`, `Queue`, `Event`, `Buffer`, `Bundle`, `Function`, BDA + Timeline) |
| `vk_device.mojo` | Mojo `VulkanDeviceContext`, `VulkanQueue`, `VulkanEvent`, `VulkanBuffer`, `VulkanBundle`, `VulkanFunction` |
| `kernel.mlir` / `bench_mlir.mlir` | MLIR `gpu.func @saxpy` kernels |
| `matmul_mlir.mlir` | MLIR `gpu.func @matmul_tiled` (512×512, 16×16 Workgroup LDS + `gpu.barrier`) |
| `bench_glsl.comp` / `matmul_glsl.comp` | Equivalent GLSL compute shaders (`glslc -O`) |
| `saxpy_bda.comp` | Descriptorless `PhysicalStorageBuffer64` + 32-byte push-constant SAXPY (`alpha`, `N` at runtime) |
| `bench_hal_c.c` | Portable C HAL test & benchmark runner |
| `bench_vulkan.c` | Pure-C Vulkan baseline benchmark |
| `run_mojo.mojo` / `bench_mojo.mojo` / `bench_bda_timeline.mojo` | Mojo host drivers & benchmarks |
| `run_all.sh` | One-command portable build, validation, and benchmark script |
