#include "vk_hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define VK_HAL_MAX_BATCH_DISPATCHES 256u
#define VK_HAL_MAX_BINDINGS_PER_DISPATCH 8u
#define VK_HAL_CMD_RING_SIZE 4u
#define VK_HAL_MAX_PENDING_WAITS 8u

#define VK_CHECK_RET(call, ret_val)                                            \
  do {                                                                         \
    VkResult err_ = (call);                                                    \
    if (err_ != VK_SUCCESS) {                                                  \
      fprintf(stderr, "[Vulkan HAL] error %d at %s:%d\n", (int)err_, __FILE__, \
              __LINE__);                                                       \
      return (ret_val);                                                        \
    }                                                                          \
  } while (0)

struct VkHalContext {
  VkInstance instance;
  VkPhysicalDevice phys;
  VkDevice device;
  uint32_t compute_qf;
  VkQueue queue;
  VkPipelineCache pipeline_cache;
  VkCommandPool xfer_pool;
  VkCommandBuffer xfer_cmd;
  VkFence xfer_fence;
};

struct VkHalQueue {
  VkHalContext *ctx;
  VkCommandPool cmd_pool;
  // Ring of command buffers + per-slot completion timeline values for safe
  // async multi-submit pipelining without stalling the host
  VkCommandBuffer cmd_bufs[VK_HAL_CMD_RING_SIZE];
  uint64_t slot_done_value[VK_HAL_CMD_RING_SIZE];
  uint32_t cur_slot;

  VkDescriptorPool dpool;
  VkSemaphore timeline_sem;
  uint64_t timeline_value;

  // Pending GPU-side waits (from vk_hal_queue_wait_event) for next submit
  VkSemaphore wait_sems[VK_HAL_MAX_PENDING_WAITS];
  uint64_t wait_vals[VK_HAL_MAX_PENDING_WAITS];
  uint32_t wait_count;

  int recording;
  uint32_t pending_dispatches;
};

struct VkHalEvent {
  VkHalContext *ctx;
  VkSemaphore timeline_sem; // Borrowed from recording queue
  uint64_t signal_value;
};

struct VkHalBuffer {
  VkHalContext *ctx;
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
  uint64_t device_address;
  int is_device_local;
};

struct VkHalBundle {
  VkHalContext *ctx;
  VkShaderModule shader;
};

struct VkHalFunction {
  VkHalContext *ctx;
  VkDescriptorSetLayout dsl;
  VkPipelineLayout pipe_layout;
  VkPipeline pipeline;
  uint32_t binding_count;
  uint32_t push_constant_size;
  int is_bda;
};

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_filter,
                                 VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((type_filter & (1u << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  return UINT32_MAX;
}

static int create_raw_buffer(VkHalContext *ctx, VkDeviceSize size,
                             VkBufferUsageFlags usage,
                             VkMemoryPropertyFlags preferred_props,
                             VkMemoryPropertyFlags fallback_props,
                             int enable_bda, VkBuffer *out_buf,
                             VkDeviceMemory *out_mem,
                             int *out_used_preferred) {
  VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  VK_CHECK_RET(vkCreateBuffer(ctx->device, &bci, NULL, out_buf), -1);

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx->device, *out_buf, &req);

  uint32_t mem_type =
      find_memory_type(ctx->phys, req.memoryTypeBits, preferred_props);
  int used_preferred = 1;
  if (mem_type == UINT32_MAX && fallback_props != 0) {
    mem_type = find_memory_type(ctx->phys, req.memoryTypeBits, fallback_props);
    used_preferred = 0;
  }
  if (mem_type == UINT32_MAX) {
    fprintf(stderr, "[Vulkan HAL] No matching memory type for buffer\n");
    vkDestroyBuffer(ctx->device, *out_buf, NULL);
    *out_buf = VK_NULL_HANDLE;
    return -1;
  }

  VkMemoryAllocateFlagsInfo flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = enable_bda ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0,
  };
  VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = enable_bda ? &flags_info : NULL,
      .allocationSize = req.size,
      .memoryTypeIndex = mem_type,
  };
  VkResult rc = vkAllocateMemory(ctx->device, &mai, NULL, out_mem);
  if (rc != VK_SUCCESS) {
    vkDestroyBuffer(ctx->device, *out_buf, NULL);
    *out_buf = VK_NULL_HANDLE;
    return (int)rc;
  }

  rc = vkBindBufferMemory(ctx->device, *out_buf, *out_mem, 0);
  if (rc != VK_SUCCESS) {
    vkFreeMemory(ctx->device, *out_mem, NULL);
    vkDestroyBuffer(ctx->device, *out_buf, NULL);
    *out_buf = VK_NULL_HANDLE;
    *out_mem = VK_NULL_HANDLE;
    return (int)rc;
  }

  if (out_used_preferred)
    *out_used_preferred = used_preferred;
  return 0;
}

static int submit_xfer_and_wait(VkHalContext *ctx) {
  VK_CHECK_RET(vkResetFences(ctx->device, 1, &ctx->xfer_fence), -1);
  VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &ctx->xfer_cmd,
  };
  VK_CHECK_RET(vkQueueSubmit(ctx->queue, 1, &si, ctx->xfer_fence), -1);
  VK_CHECK_RET(vkWaitForFences(ctx->device, 1, &ctx->xfer_fence, VK_TRUE,
                               UINT64_MAX),
               -1);
  return 0;
}

static int wait_timeline(VkHalContext *ctx, VkSemaphore sem, uint64_t value) {
  if (value == 0)
    return 0;
  VkSemaphoreWaitInfo swi = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &sem,
      .pValues = &value,
  };
  VK_CHECK_RET(vkWaitSemaphores(ctx->device, &swi, UINT64_MAX), -1);
  return 0;
}

static uint8_t *read_file(const char *path, size_t *out_size) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0) {
    fclose(f);
    return NULL;
  }
  uint8_t *data = (uint8_t *)malloc((size_t)sz);
  if (!data || fread(data, 1, (size_t)sz, f) != (size_t)sz) {
    free(data);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *out_size = (size_t)sz;
  return data;
}

// ---------------------------------------------------------------------------
// Context lifecycle (Vulkan 1.3 + BDA + Timeline Semaphores)
// ---------------------------------------------------------------------------

VkHalContext *vk_hal_context_create(uint32_t device_index) {
  VkHalContext *ctx = (VkHalContext *)calloc(1, sizeof(VkHalContext));
  if (!ctx)
    return NULL;

  VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "MLIR-Vulkan-HAL",
      .applicationVersion = VK_MAKE_VERSION(4, 0, 0),
      .apiVersion = VK_API_VERSION_1_3,
  };
  VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
  };
  if (vkCreateInstance(&ici, NULL, &ctx->instance) != VK_SUCCESS) {
    free(ctx);
    return NULL;
  }

  uint32_t gpu_count = 0;
  if (vkEnumeratePhysicalDevices(ctx->instance, &gpu_count, NULL) !=
          VK_SUCCESS ||
      gpu_count == 0) {
    fprintf(stderr, "[Vulkan HAL] No Vulkan physical devices found\n");
    vk_hal_context_destroy(ctx);
    return NULL;
  }
  VkPhysicalDevice *gpus =
      (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * gpu_count);
  if (!gpus || vkEnumeratePhysicalDevices(ctx->instance, &gpu_count, gpus) !=
                   VK_SUCCESS) {
    free(gpus);
    vk_hal_context_destroy(ctx);
    return NULL;
  }
  if (device_index >= gpu_count)
    device_index = 0;
  ctx->phys = gpus[device_index];
  free(gpus);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(ctx->phys, &props);
  printf("[Vulkan HAL] Context ready: %s (Vulkan %u.%u.%u | BDA + Timeline)\n",
         props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
         VK_API_VERSION_MINOR(props.apiVersion),
         VK_API_VERSION_PATCH(props.apiVersion));

  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &qf_count, NULL);
  VkQueueFamilyProperties *qfs = (VkQueueFamilyProperties *)malloc(
      sizeof(VkQueueFamilyProperties) * qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &qf_count, qfs);
  ctx->compute_qf = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; i++) {
    if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      ctx->compute_qf = i;
      break;
    }
  }
  if (ctx->compute_qf == UINT32_MAX) {
    for (uint32_t i = 0; i < qf_count; i++) {
      if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        ctx->compute_qf = i;
        break;
      }
    }
  }
  free(qfs);
  if (ctx->compute_qf == UINT32_MAX) {
    fprintf(stderr, "[Vulkan HAL] No compute queue family found\n");
    vk_hal_context_destroy(ctx);
    return NULL;
  }

  VkPhysicalDeviceVulkan12Features vk12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .timelineSemaphore = VK_TRUE,
      .bufferDeviceAddress = VK_TRUE,
  };
  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = ctx->compute_qf,
      .queueCount = 1,
      .pQueuePriorities = &priority,
  };
  VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &vk12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
  };
  if (vkCreateDevice(ctx->phys, &dci, NULL, &ctx->device) != VK_SUCCESS) {
    fprintf(stderr, "[Vulkan HAL] Failed to create device with BDA+Timeline\n");
    vk_hal_context_destroy(ctx);
    return NULL;
  }
  vkGetDeviceQueue(ctx->device, ctx->compute_qf, 0, &ctx->queue);

  VkPipelineCacheCreateInfo pcci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
  };
  if (vkCreatePipelineCache(ctx->device, &pcci, NULL, &ctx->pipeline_cache) !=
      VK_SUCCESS) {
    vk_hal_context_destroy(ctx);
    return NULL;
  }

  VkCommandPoolCreateInfo xpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = ctx->compute_qf,
  };
  if (vkCreateCommandPool(ctx->device, &xpci, NULL, &ctx->xfer_pool) !=
      VK_SUCCESS) {
    vk_hal_context_destroy(ctx);
    return NULL;
  }

  VkCommandBufferAllocateInfo xbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = ctx->xfer_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  if (vkAllocateCommandBuffers(ctx->device, &xbai, &ctx->xfer_cmd) !=
      VK_SUCCESS) {
    vk_hal_context_destroy(ctx);
    return NULL;
  }

  VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(ctx->device, &fci, NULL, &ctx->xfer_fence) != VK_SUCCESS) {
    vk_hal_context_destroy(ctx);
    return NULL;
  }

  return ctx;
}

void vk_hal_context_destroy(VkHalContext *ctx) {
  if (!ctx)
    return;
  if (ctx->device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(ctx->device);
    if (ctx->xfer_fence != VK_NULL_HANDLE)
      vkDestroyFence(ctx->device, ctx->xfer_fence, NULL);
    if (ctx->xfer_pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(ctx->device, ctx->xfer_pool, NULL);
    if (ctx->pipeline_cache != VK_NULL_HANDLE)
      vkDestroyPipelineCache(ctx->device, ctx->pipeline_cache, NULL);
    vkDestroyDevice(ctx->device, NULL);
  }
  if (ctx->instance != VK_NULL_HANDLE)
    vkDestroyInstance(ctx->instance, NULL);
  free(ctx);
}

uint64_t vk_hal_context_pipeline_cache_size(VkHalContext *ctx) {
  if (!ctx || ctx->pipeline_cache == VK_NULL_HANDLE)
    return 0;
  size_t sz = 0;
  if (vkGetPipelineCacheData(ctx->device, ctx->pipeline_cache, &sz, NULL) !=
      VK_SUCCESS) {
    return 0;
  }
  return (uint64_t)sz;
}

// ---------------------------------------------------------------------------
// Queue (batched async command stream + timeline semaphore + BDA dispatch)
// ---------------------------------------------------------------------------

static int queue_ensure_recording(VkHalQueue *q) {
  if (q->recording)
    return 0;
  // Wait for the current ring slot's previous submission (if any) before reuse
  if (q->slot_done_value[q->cur_slot] != 0) {
    if (wait_timeline(q->ctx, q->timeline_sem,
                      q->slot_done_value[q->cur_slot]) != 0) {
      return -1;
    }
  }
  VkCommandBuffer cmd = q->cmd_bufs[q->cur_slot];
  VK_CHECK_RET(vkResetCommandBuffer(cmd, 0), -1);
  VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  VK_CHECK_RET(vkBeginCommandBuffer(cmd, &cbbi), -1);
  q->recording = 1;
  return 0;
}

static void queue_insert_compute_barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
  };
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                       NULL, 0, NULL);
}

// Submit the current recorded batch asynchronously, signaling timeline_value+1.
// Returns the signaled timeline value (or 0 on error).
static uint64_t queue_flush_async(VkHalQueue *q) {
  if (!q->recording)
    return q->timeline_value;

  VkCommandBuffer cmd = q->cmd_bufs[q->cur_slot];
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    return 0;

  uint64_t sig_val = ++q->timeline_value;
  VkPipelineStageFlags wait_stages[VK_HAL_MAX_PENDING_WAITS];
  for (uint32_t i = 0; i < q->wait_count; i++) {
    wait_stages[i] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }

  VkTimelineSemaphoreSubmitInfo tsi = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .waitSemaphoreValueCount = q->wait_count,
      .pWaitSemaphoreValues = q->wait_count ? q->wait_vals : NULL,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &sig_val,
  };
  VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = &tsi,
      .waitSemaphoreCount = q->wait_count,
      .pWaitSemaphores = q->wait_count ? q->wait_sems : NULL,
      .pWaitDstStageMask = q->wait_count ? wait_stages : NULL,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &q->timeline_sem,
  };
  if (vkQueueSubmit(q->ctx->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
    return 0;
  }

  q->slot_done_value[q->cur_slot] = sig_val;
  q->cur_slot = (q->cur_slot + 1u) % VK_HAL_CMD_RING_SIZE;
  q->wait_count = 0;
  q->recording = 0;
  q->pending_dispatches = 0;
  return sig_val;
}

VkHalQueue *vk_hal_queue_create(VkHalContext *ctx) {
  if (!ctx)
    return NULL;
  VkHalQueue *q = (VkHalQueue *)calloc(1, sizeof(VkHalQueue));
  if (!q)
    return NULL;
  q->ctx = ctx;

  VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = ctx->compute_qf,
  };
  if (vkCreateCommandPool(ctx->device, &cpci, NULL, &q->cmd_pool) !=
      VK_SUCCESS) {
    vk_hal_queue_destroy(q);
    return NULL;
  }

  VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = q->cmd_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = VK_HAL_CMD_RING_SIZE,
  };
  if (vkAllocateCommandBuffers(ctx->device, &cbai, q->cmd_bufs) != VK_SUCCESS) {
    vk_hal_queue_destroy(q);
    return NULL;
  }

  VkSemaphoreTypeCreateInfo stci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
  };
  VkSemaphoreCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &stci,
  };
  if (vkCreateSemaphore(ctx->device, &sci, NULL, &q->timeline_sem) !=
      VK_SUCCESS) {
    vk_hal_queue_destroy(q);
    return NULL;
  }

  VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount =
          VK_HAL_MAX_BATCH_DISPATCHES * VK_HAL_MAX_BINDINGS_PER_DISPATCH,
  };
  VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = VK_HAL_MAX_BATCH_DISPATCHES,
      .poolSizeCount = 1,
      .pPoolSizes = &dps,
  };
  if (vkCreateDescriptorPool(ctx->device, &dpci, NULL, &q->dpool) !=
      VK_SUCCESS) {
    vk_hal_queue_destroy(q);
    return NULL;
  }

  return q;
}

void vk_hal_queue_destroy(VkHalQueue *q) {
  if (!q)
    return;
  vk_hal_queue_synchronize(q);
  if (q->ctx && q->ctx->device != VK_NULL_HANDLE) {
    if (q->dpool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(q->ctx->device, q->dpool, NULL);
    if (q->timeline_sem != VK_NULL_HANDLE)
      vkDestroySemaphore(q->ctx->device, q->timeline_sem, NULL);
    if (q->cmd_pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(q->ctx->device, q->cmd_pool, NULL);
  }
  free(q);
}

int32_t vk_hal_queue_execute(VkHalQueue *q, VkHalFunction *fn,
                             VkHalBuffer *const *buffers,
                             uint32_t buffer_count, uint32_t groups_x,
                             uint32_t groups_y, uint32_t groups_z) {
  if (!q || !fn || fn->is_bda || !buffers ||
      buffer_count != fn->binding_count ||
      buffer_count > VK_HAL_MAX_BINDINGS_PER_DISPATCH) {
    return -1;
  }
  VkHalContext *ctx = q->ctx;

  if (q->pending_dispatches >= VK_HAL_MAX_BATCH_DISPATCHES) {
    int32_t rc = vk_hal_queue_synchronize(q);
    if (rc != 0)
      return rc;
  }

  if (queue_ensure_recording(q) != 0)
    return -1;

  VkCommandBuffer cmd = q->cmd_bufs[q->cur_slot];
  if (q->pending_dispatches > 0)
    queue_insert_compute_barrier(cmd);

  VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = q->dpool,
      .descriptorSetCount = 1,
      .pSetLayouts = &fn->dsl,
  };
  VkDescriptorSet dset = VK_NULL_HANDLE;
  VK_CHECK_RET(vkAllocateDescriptorSets(ctx->device, &dsai, &dset), -1);

  VkDescriptorBufferInfo dbis[VK_HAL_MAX_BINDINGS_PER_DISPATCH];
  VkWriteDescriptorSet writes[VK_HAL_MAX_BINDINGS_PER_DISPATCH];
  for (uint32_t i = 0; i < buffer_count; i++) {
    dbis[i] = (VkDescriptorBufferInfo){
        .buffer = buffers[i]->buffer,
        .offset = 0,
        .range = buffers[i]->size,
    };
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dset,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbis[i],
    };
  }
  vkUpdateDescriptorSets(ctx->device, buffer_count, writes, 0, NULL);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fn->pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fn->pipe_layout,
                          0, 1, &dset, 0, NULL);
  vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
  q->pending_dispatches++;
  return 0;
}

int32_t vk_hal_queue_execute_bda(VkHalQueue *q, VkHalFunction *fn,
                                 const void *push_data, uint32_t push_size,
                                 uint32_t groups_x, uint32_t groups_y,
                                 uint32_t groups_z) {
  if (!q || !fn || !fn->is_bda || !push_data ||
      push_size != fn->push_constant_size) {
    return -1;
  }

  if (queue_ensure_recording(q) != 0)
    return -1;

  VkCommandBuffer cmd = q->cmd_bufs[q->cur_slot];
  if (q->pending_dispatches > 0)
    queue_insert_compute_barrier(cmd);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fn->pipeline);
  vkCmdPushConstants(cmd, fn->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     push_size, push_data);
  vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
  q->pending_dispatches++;
  return 0;
}

int32_t vk_hal_queue_synchronize(VkHalQueue *q) {
  if (!q)
    return -1;
  if (q->recording) {
    if (queue_flush_async(q) == 0)
      return -1;
  }
  if (q->timeline_value > 0) {
    if (wait_timeline(q->ctx, q->timeline_sem, q->timeline_value) != 0)
      return -1;
  }
  VK_CHECK_RET(vkResetDescriptorPool(q->ctx->device, q->dpool, 0), -1);
  return 0;
}

// ---------------------------------------------------------------------------
// Timeline-Semaphore Events (M_driver_event)
// ---------------------------------------------------------------------------

VkHalEvent *vk_hal_event_create(VkHalContext *ctx) {
  if (!ctx)
    return NULL;
  VkHalEvent *ev = (VkHalEvent *)calloc(1, sizeof(VkHalEvent));
  if (!ev)
    return NULL;
  ev->ctx = ctx;
  return ev;
}

void vk_hal_event_destroy(VkHalEvent *ev) { free(ev); }

int32_t vk_hal_queue_record_event(VkHalQueue *q, VkHalEvent *ev) {
  if (!q || !ev)
    return -1;
  uint64_t val = q->recording ? queue_flush_async(q) : q->timeline_value;
  if (q->recording && val == 0)
    return -1;
  ev->timeline_sem = q->timeline_sem;
  ev->signal_value = val;
  return 0;
}

int32_t vk_hal_queue_wait_event(VkHalQueue *q, VkHalEvent *ev) {
  if (!q || !ev)
    return -1;
  if (ev->timeline_sem == VK_NULL_HANDLE || ev->signal_value == 0)
    return 0;
  // If commands are already being recorded, flush them first so the wait
  // applies cleanly to subsequent commands on this queue
  if (q->recording) {
    if (queue_flush_async(q) == 0)
      return -1;
  }
  if (q->wait_count >= VK_HAL_MAX_PENDING_WAITS)
    return -1;
  q->wait_sems[q->wait_count] = ev->timeline_sem;
  q->wait_vals[q->wait_count] = ev->signal_value;
  q->wait_count++;
  return 0;
}

int32_t vk_hal_event_synchronize(VkHalEvent *ev) {
  if (!ev)
    return -1;
  if (ev->timeline_sem == VK_NULL_HANDLE || ev->signal_value == 0)
    return 0;
  return wait_timeline(ev->ctx, ev->timeline_sem, ev->signal_value);
}

int32_t vk_hal_event_is_ready(VkHalEvent *ev) {
  if (!ev)
    return 0;
  if (ev->timeline_sem == VK_NULL_HANDLE || ev->signal_value == 0)
    return 1;
  uint64_t cur = 0;
  if (vkGetSemaphoreCounterValue(ev->ctx->device, ev->timeline_sem, &cur) !=
      VK_SUCCESS) {
    return 0;
  }
  return cur >= ev->signal_value ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Device-local memory + 64-bit device address + staging transfers
// ---------------------------------------------------------------------------

VkHalBuffer *vk_hal_buffer_create(VkHalContext *ctx, uint64_t size_bytes) {
  if (!ctx || size_bytes == 0)
    return NULL;
  VkHalBuffer *buf = (VkHalBuffer *)calloc(1, sizeof(VkHalBuffer));
  if (!buf)
    return NULL;

  buf->ctx = ctx;
  buf->size = (VkDeviceSize)size_bytes;

  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  if (create_raw_buffer(ctx, buf->size, usage,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        1, &buf->buffer, &buf->memory,
                        &buf->is_device_local) != 0) {
    free(buf);
    return NULL;
  }

  VkBufferDeviceAddressInfo bdai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = buf->buffer,
  };
  buf->device_address = vkGetBufferDeviceAddress(ctx->device, &bdai);
  return buf;
}

void vk_hal_buffer_destroy(VkHalBuffer *buf) {
  if (!buf)
    return;
  if (buf->ctx && buf->ctx->device != VK_NULL_HANDLE) {
    if (buf->buffer != VK_NULL_HANDLE)
      vkDestroyBuffer(buf->ctx->device, buf->buffer, NULL);
    if (buf->memory != VK_NULL_HANDLE)
      vkFreeMemory(buf->ctx->device, buf->memory, NULL);
  }
  free(buf);
}

uint64_t vk_hal_buffer_device_address(VkHalBuffer *buf) {
  return buf ? buf->device_address : 0;
}

int32_t vk_hal_buffer_upload(VkHalBuffer *buf, const void *src,
                             uint64_t size_bytes) {
  if (!buf || !src || size_bytes == 0 || size_bytes > buf->size)
    return -1;
  VkHalContext *ctx = buf->ctx;

  VkBuffer staging_buf = VK_NULL_HANDLE;
  VkDeviceMemory staging_mem = VK_NULL_HANDLE;
  int rc = create_raw_buffer(ctx, (VkDeviceSize)size_bytes,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             0, 0, &staging_buf, &staging_mem, NULL);
  if (rc != 0)
    return rc;

  void *mapped = NULL;
  VK_CHECK_RET(
      vkMapMemory(ctx->device, staging_mem, 0, size_bytes, 0, &mapped), -1);
  memcpy(mapped, src, (size_t)size_bytes);
  vkUnmapMemory(ctx->device, staging_mem);

  VK_CHECK_RET(vkResetCommandBuffer(ctx->xfer_cmd, 0), -1);
  VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  VK_CHECK_RET(vkBeginCommandBuffer(ctx->xfer_cmd, &cbbi), -1);
  VkBufferCopy region = {.srcOffset = 0, .dstOffset = 0, .size = size_bytes};
  vkCmdCopyBuffer(ctx->xfer_cmd, staging_buf, buf->buffer, 1, &region);
  VK_CHECK_RET(vkEndCommandBuffer(ctx->xfer_cmd), -1);

  rc = submit_xfer_and_wait(ctx);

  vkDestroyBuffer(ctx->device, staging_buf, NULL);
  vkFreeMemory(ctx->device, staging_mem, NULL);
  return rc;
}

int32_t vk_hal_buffer_download(VkHalBuffer *buf, void *dst,
                               uint64_t size_bytes) {
  if (!buf || !dst || size_bytes == 0 || size_bytes > buf->size)
    return -1;
  VkHalContext *ctx = buf->ctx;

  VkBuffer staging_buf = VK_NULL_HANDLE;
  VkDeviceMemory staging_mem = VK_NULL_HANDLE;
  int rc = create_raw_buffer(ctx, (VkDeviceSize)size_bytes,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             0, 0, &staging_buf, &staging_mem, NULL);
  if (rc != 0)
    return rc;

  VK_CHECK_RET(vkResetCommandBuffer(ctx->xfer_cmd, 0), -1);
  VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  VK_CHECK_RET(vkBeginCommandBuffer(ctx->xfer_cmd, &cbbi), -1);
  VkBufferCopy region = {.srcOffset = 0, .dstOffset = 0, .size = size_bytes};
  vkCmdCopyBuffer(ctx->xfer_cmd, buf->buffer, staging_buf, 1, &region);
  VK_CHECK_RET(vkEndCommandBuffer(ctx->xfer_cmd), -1);

  rc = submit_xfer_and_wait(ctx);
  if (rc == 0) {
    void *mapped = NULL;
    if (vkMapMemory(ctx->device, staging_mem, 0, size_bytes, 0, &mapped) ==
        VK_SUCCESS) {
      memcpy(dst, mapped, (size_t)size_bytes);
      vkUnmapMemory(ctx->device, staging_mem);
    } else {
      rc = -1;
    }
  }

  vkDestroyBuffer(ctx->device, staging_buf, NULL);
  vkFreeMemory(ctx->device, staging_mem, NULL);
  return rc;
}

// ---------------------------------------------------------------------------
// Bundle & Function (descriptor-set or descriptorless BDA + push constants)
// ---------------------------------------------------------------------------

VkHalBundle *vk_hal_bundle_load(VkHalContext *ctx, const char *spv_path) {
  if (!ctx || !spv_path)
    return NULL;
  size_t spv_size = 0;
  uint8_t *spv_code = read_file(spv_path, &spv_size);
  if (!spv_code)
    return NULL;

  VkHalBundle *b = (VkHalBundle *)calloc(1, sizeof(VkHalBundle));
  if (!b) {
    free(spv_code);
    return NULL;
  }
  b->ctx = ctx;

  VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size,
      .pCode = (const uint32_t *)spv_code,
  };
  VkResult rc = vkCreateShaderModule(ctx->device, &smci, NULL, &b->shader);
  free(spv_code);
  if (rc != VK_SUCCESS) {
    free(b);
    return NULL;
  }
  return b;
}

void vk_hal_bundle_unload(VkHalBundle *bundle) {
  if (!bundle)
    return;
  if (bundle->ctx && bundle->ctx->device != VK_NULL_HANDLE &&
      bundle->shader != VK_NULL_HANDLE) {
    vkDestroyShaderModule(bundle->ctx->device, bundle->shader, NULL);
  }
  free(bundle);
}

VkHalFunction *vk_hal_function_load(VkHalBundle *bundle,
                                    const char *entry_point,
                                    uint32_t binding_count) {
  if (!bundle || !entry_point || binding_count == 0 ||
      binding_count > VK_HAL_MAX_BINDINGS_PER_DISPATCH) {
    return NULL;
  }
  VkHalContext *ctx = bundle->ctx;
  VkHalFunction *fn = (VkHalFunction *)calloc(1, sizeof(VkHalFunction));
  if (!fn)
    return NULL;
  fn->ctx = ctx;
  fn->binding_count = binding_count;
  fn->is_bda = 0;

  VkDescriptorSetLayoutBinding bindings[VK_HAL_MAX_BINDINGS_PER_DISPATCH];
  for (uint32_t i = 0; i < binding_count; i++) {
    bindings[i] = (VkDescriptorSetLayoutBinding){
        .binding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = binding_count,
      .pBindings = bindings,
  };
  if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &fn->dsl) !=
      VK_SUCCESS) {
    vk_hal_function_unload(fn);
    return NULL;
  }

  VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &fn->dsl,
  };
  if (vkCreatePipelineLayout(ctx->device, &plci, NULL, &fn->pipe_layout) !=
      VK_SUCCESS) {
    vk_hal_function_unload(fn);
    return NULL;
  }

  VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = bundle->shader,
              .pName = entry_point,
          },
      .layout = fn->pipe_layout,
  };
  if (vkCreateComputePipelines(ctx->device, ctx->pipeline_cache, 1, &cpci, NULL,
                               &fn->pipeline) != VK_SUCCESS) {
    vk_hal_function_unload(fn);
    return NULL;
  }
  return fn;
}

VkHalFunction *vk_hal_function_load_bda(VkHalBundle *bundle,
                                        const char *entry_point,
                                        uint32_t push_constant_size) {
  if (!bundle || !entry_point || push_constant_size == 0)
    return NULL;
  VkHalContext *ctx = bundle->ctx;
  VkHalFunction *fn = (VkHalFunction *)calloc(1, sizeof(VkHalFunction));
  if (!fn)
    return NULL;
  fn->ctx = ctx;
  fn->push_constant_size = push_constant_size;
  fn->is_bda = 1;

  VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = push_constant_size,
  };
  VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 0, // Zero descriptor sets!
      .pSetLayouts = NULL,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
  };
  if (vkCreatePipelineLayout(ctx->device, &plci, NULL, &fn->pipe_layout) !=
      VK_SUCCESS) {
    vk_hal_function_unload(fn);
    return NULL;
  }

  VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = bundle->shader,
              .pName = entry_point,
          },
      .layout = fn->pipe_layout,
  };
  if (vkCreateComputePipelines(ctx->device, ctx->pipeline_cache, 1, &cpci, NULL,
                               &fn->pipeline) != VK_SUCCESS) {
    vk_hal_function_unload(fn);
    return NULL;
  }
  return fn;
}

void vk_hal_function_unload(VkHalFunction *fn) {
  if (!fn)
    return;
  if (fn->ctx && fn->ctx->device != VK_NULL_HANDLE) {
    if (fn->pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(fn->ctx->device, fn->pipeline, NULL);
    if (fn->pipe_layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(fn->ctx->device, fn->pipe_layout, NULL);
    if (fn->dsl != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(fn->ctx->device, fn->dsl, NULL);
  }
  free(fn);
}
