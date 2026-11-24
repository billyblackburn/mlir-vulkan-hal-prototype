#ifndef VK_HAL_H
#define VK_HAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles aligned with Modular's SDK/include/HAL model:
//   Context  -> M_driver_context  (Vulkan 1.3 + BDA + Timeline Semaphores)
//   Queue    -> M_driver_queue    (batched async command stream + barriers)
//   Event    -> M_driver_event    (timeline semaphore synchronization point)
//   Buffer   -> M_driver_memory   (DEVICE_LOCAL + 64-bit GPU device address)
//   Bundle   -> M_driver_bundle   (loaded SPIR-V module)
//   Function -> M_driver_function (descriptor or descriptorless BDA pipeline)
typedef struct VkHalContext VkHalContext;
typedef struct VkHalQueue VkHalQueue;
typedef struct VkHalEvent VkHalEvent;
typedef struct VkHalBuffer VkHalBuffer;
typedef struct VkHalBundle VkHalBundle;
typedef struct VkHalFunction VkHalFunction;

// Context lifecycle
VkHalContext *vk_hal_context_create(uint32_t device_index);
void vk_hal_context_destroy(VkHalContext *ctx);
uint64_t vk_hal_context_pipeline_cache_size(VkHalContext *ctx);

// Queue (async stream with batched command recording + timeline semaphore)
VkHalQueue *vk_hal_queue_create(VkHalContext *ctx);
void vk_hal_queue_destroy(VkHalQueue *q);

// Classic descriptor-set dispatch
int32_t vk_hal_queue_execute(VkHalQueue *q, VkHalFunction *fn,
                             VkHalBuffer *const *buffers,
                             uint32_t buffer_count, uint32_t groups_x,
                             uint32_t groups_y, uint32_t groups_z);

// Descriptorless BDA + push-constant dispatch (raw 64-bit pointers + scalars)
int32_t vk_hal_queue_execute_bda(VkHalQueue *q, VkHalFunction *fn,
                                 const void *push_data, uint32_t push_size,
                                 uint32_t groups_x, uint32_t groups_y,
                                 uint32_t groups_z);

int32_t vk_hal_queue_synchronize(VkHalQueue *q);

// Timeline-semaphore events (M_driver_event / queue_record_event / wait)
VkHalEvent *vk_hal_event_create(VkHalContext *ctx);
void vk_hal_event_destroy(VkHalEvent *ev);
int32_t vk_hal_queue_record_event(VkHalQueue *q, VkHalEvent *ev);
int32_t vk_hal_queue_wait_event(VkHalQueue *q, VkHalEvent *ev);
int32_t vk_hal_event_synchronize(VkHalEvent *ev);
int32_t vk_hal_event_is_ready(VkHalEvent *ev);

// Device-local memory lifecycle + 64-bit device address + staging transfers
VkHalBuffer *vk_hal_buffer_create(VkHalContext *ctx, uint64_t size_bytes);
void vk_hal_buffer_destroy(VkHalBuffer *buf);
uint64_t vk_hal_buffer_device_address(VkHalBuffer *buf);
int32_t vk_hal_buffer_upload(VkHalBuffer *buf, const void *src,
                             uint64_t size_bytes);
int32_t vk_hal_buffer_download(VkHalBuffer *buf, void *dst,
                               uint64_t size_bytes);

// Bundle (SPIR-V module) & Function (descriptor or BDA/push-constant pipeline)
VkHalBundle *vk_hal_bundle_load(VkHalContext *ctx, const char *spv_path);
void vk_hal_bundle_unload(VkHalBundle *bundle);

VkHalFunction *vk_hal_function_load(VkHalBundle *bundle,
                                    const char *entry_point,
                                    uint32_t binding_count);
VkHalFunction *vk_hal_function_load_bda(VkHalBundle *bundle,
                                        const char *entry_point,
                                        uint32_t push_constant_size);
void vk_hal_function_unload(VkHalFunction *fn);

#ifdef __cplusplus
}
#endif

#endif // VK_HAL_H
