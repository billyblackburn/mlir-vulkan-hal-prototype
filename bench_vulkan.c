#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define N_ELEMS (4u * 1024u * 1024u)
#define WG_SIZE 256u
#define GRID_X (N_ELEMS / WG_SIZE)
#define WARMUP_ITERS 10
#define BENCH_ITERS 100

#define VK_CHECK(call)                                                         \
  do {                                                                         \
    VkResult e_ = (call);                                                      \
    if (e_ != VK_SUCCESS) {                                                    \
      fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)e_, __FILE__,         \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static uint32_t find_mem(VkPhysicalDevice phys, uint32_t bits,
                         VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  }
  return UINT32_MAX;
}

static void make_buf(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize sz,
                     VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                     VkBuffer *b, VkDeviceMemory *m) {
  VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sz,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  VK_CHECK(vkCreateBuffer(dev, &bci, NULL, b));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, *b, &req);
  uint32_t mt = find_mem(phys, req.memoryTypeBits, props);
  if (mt == UINT32_MAX) {
    fprintf(stderr, "No memory type found\n");
    exit(1);
  }
  VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = mt,
  };
  VK_CHECK(vkAllocateMemory(dev, &mai, NULL, m));
  VK_CHECK(vkBindBufferMemory(dev, *b, *m, 0));
}

static uint8_t *load_file(const char *path, size_t *out_sz) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *buf = (uint8_t *)malloc((size_t)sz);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz)
    exit(1);
  fclose(f);
  *out_sz = (size_t)sz;
  return buf;
}

static void copy_buf(VkDevice dev, VkQueue q, VkCommandBuffer cmd, VkFence f,
                     VkBuffer src, VkBuffer dst, VkDeviceSize sz) {
  VK_CHECK(vkResetCommandBuffer(cmd, 0));
  VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
  VkBufferCopy r = {.size = sz};
  vkCmdCopyBuffer(cmd, src, dst, 1, &r);
  VK_CHECK(vkEndCommandBuffer(cmd));
  VK_CHECK(vkResetFences(dev, 1, &f));
  VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
  };
  VK_CHECK(vkQueueSubmit(q, 1, &si, f));
  VK_CHECK(vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX));
}

static void run_shader_bench(VkDevice dev, VkQueue q, VkCommandBuffer cmd,
                             VkFence fence, VkPipelineCache cache,
                             VkBuffer d_a, VkBuffer d_b, VkBuffer d_c,
                             VkBuffer d_d, VkDeviceSize buf_sz,
                             VkBuffer staging, VkDeviceMemory staging_mem,
                             const float *h_a, const float *h_b,
                             const char *label, const char *spv_path,
                             const char *entry) {
  size_t spv_sz = 0;
  uint8_t *spv = load_file(spv_path, &spv_sz);
  VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_sz,
      .pCode = (const uint32_t *)spv,
  };
  VkShaderModule sm;
  VK_CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
  free(spv);

  VkDescriptorSetLayoutBinding bindings[3];
  for (uint32_t i = 0; i < 3; i++) {
    bindings[i] = (VkDescriptorSetLayoutBinding){
        .binding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = bindings,
  };
  VkDescriptorSetLayout dsl;
  VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

  VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &dsl,
  };
  VkPipelineLayout pl;
  VK_CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

  VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = sm,
              .pName = entry,
          },
      .layout = pl,
  };
  VkPipeline pipe;
  VK_CHECK(vkCreateComputePipelines(dev, cache, 1, &cpci, NULL, &pipe));

  VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 6,
  };
  VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2,
      .poolSizeCount = 1,
      .pPoolSizes = &dps,
  };
  VkDescriptorPool dpool;
  VK_CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));

  VkDescriptorSetLayout layouts[2] = {dsl, dsl};
  VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool,
      .descriptorSetCount = 2,
      .pSetLayouts = layouts,
  };
  VkDescriptorSet dsets[2];
  VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, dsets));

  // dsets[0]: (A, B) -> C; dsets[1]: (C, B) -> D
  VkBuffer set0_bufs[3] = {d_a, d_b, d_c};
  VkBuffer set1_bufs[3] = {d_c, d_b, d_d};
  VkDescriptorBufferInfo dbis[6];
  VkWriteDescriptorSet writes[6];
  for (uint32_t i = 0; i < 3; i++) {
    dbis[i] = (VkDescriptorBufferInfo){set0_bufs[i], 0, buf_sz};
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dsets[0],
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbis[i],
    };
    dbis[3 + i] = (VkDescriptorBufferInfo){set1_bufs[i], 0, buf_sz};
    writes[3 + i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dsets[1],
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbis[3 + i],
    };
  }
  vkUpdateDescriptorSets(dev, 6, writes, 0, NULL);

  const double bytes_per_iter = 6.0 * (double)buf_sz; // 2 passes * (2R + 1W)

  // Mode A: 2 separate submits per chained iter (matches HAL dispatch_3buf)
  for (int it = 0; it < WARMUP_ITERS + BENCH_ITERS; it++) {
    double t0 = 0.0;
    if (it == WARMUP_ITERS)
      t0 = now_ms();
    static double t_start = 0.0;
    if (it == WARMUP_ITERS)
      t_start = t0;

    for (int pass = 0; pass < 2; pass++) {
      VK_CHECK(vkResetCommandBuffer(cmd, 0));
      VkCommandBufferBeginInfo bi = {
          .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
          .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1,
                              &dsets[pass], 0, NULL);
      vkCmdDispatch(cmd, GRID_X, 1, 1);
      VK_CHECK(vkEndCommandBuffer(cmd));
      VK_CHECK(vkResetFences(dev, 1, &fence));
      VkSubmitInfo si = {
          .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
          .commandBufferCount = 1,
          .pCommandBuffers = &cmd,
      };
      VK_CHECK(vkQueueSubmit(q, 1, &si, fence));
      VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    }

    if (it == WARMUP_ITERS + BENCH_ITERS - 1) {
      double elapsed_ms = now_ms() - t_start;
      double avg_ms = elapsed_ms / BENCH_ITERS;
      double gbps = (bytes_per_iter / 1e9) / (avg_ms / 1e3);
      printf("  %-22s [C sync-per-dispatch]: %6.3f ms/iter | %6.2f GB/s\n",
             label, avg_ms, gbps);
    }
  }

  // Mode B: Batched single command buffer with compute barrier between passes
  VK_CHECK(vkResetCommandBuffer(cmd, 0));
  VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  };
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1,
                          &dsets[0], 0, NULL);
  vkCmdDispatch(cmd, GRID_X, 1, 1);
  VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
  };
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                       NULL, 0, NULL);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1,
                          &dsets[1], 0, NULL);
  vkCmdDispatch(cmd, GRID_X, 1, 1);
  VK_CHECK(vkEndCommandBuffer(cmd));

  for (int it = 0; it < WARMUP_ITERS; it++) {
    VK_CHECK(vkResetFences(dev, 1, &fence));
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    VK_CHECK(vkQueueSubmit(q, 1, &si, fence));
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  }
  double t_batch = now_ms();
  for (int it = 0; it < BENCH_ITERS; it++) {
    VK_CHECK(vkResetFences(dev, 1, &fence));
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    VK_CHECK(vkQueueSubmit(q, 1, &si, fence));
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  }
  double avg_batch = (now_ms() - t_batch) / BENCH_ITERS;
  double gbps_batch = (bytes_per_iter / 1e9) / (avg_batch / 1e3);
  printf("  %-22s [C batched+barrier  ]: %6.3f ms/iter | %6.2f GB/s\n", label,
         avg_batch, gbps_batch);

  // Verify correctness
  copy_buf(dev, q, cmd, fence, d_d, staging, buf_sz);
  float *mapped = NULL;
  VK_CHECK(vkMapMemory(dev, staging_mem, 0, buf_sz, 0, (void **)&mapped));
  float max_err = 0.0f;
  for (uint32_t i = 0; i < N_ELEMS; i++) {
    float expected = 2.5f * (2.5f * h_a[i] + h_b[i]) + h_b[i];
    float diff = fabsf(mapped[i] - expected);
    if (diff > max_err)
      max_err = diff;
  }
  vkUnmapMemory(dev, staging_mem);
  printf("  %-22s [verification       ]: max_err = %.6f\n", label, max_err);

  vkDestroyDescriptorPool(dev, dpool, NULL);
  vkDestroyPipeline(dev, pipe, NULL);
  vkDestroyPipelineLayout(dev, pl, NULL);
  vkDestroyDescriptorSetLayout(dev, dsl, NULL);
  vkDestroyShaderModule(dev, sm, NULL);
}

int main(void) {
  VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "Vulkan-SAXPY-Bench",
      .apiVersion = VK_API_VERSION_1_3,
  };
  VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
  };
  VkInstance inst;
  VK_CHECK(vkCreateInstance(&ici, NULL, &inst));

  uint32_t cnt = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(inst, &cnt, NULL));
  VkPhysicalDevice *gpus = (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * cnt);
  VK_CHECK(vkEnumeratePhysicalDevices(inst, &cnt, gpus));
  VkPhysicalDevice phys = gpus[0];
  free(gpus);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(phys, &props);
  printf("[C Vulkan Bench] Device: %s | N = %u (16.0 MiB/buf, 2-pass chained SAXPY = 96.0 MiB / %.2f MB/iter)\n",
         props.deviceName, N_ELEMS, (double)(N_ELEMS * 4 * 6) / 1e6);

  uint32_t qfc = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfc, NULL);
  VkQueueFamilyProperties *qfs = (VkQueueFamilyProperties *)malloc(sizeof(*qfs) * qfc);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfc, qfs);
  uint32_t qf = UINT32_MAX;
  for (uint32_t i = 0; i < qfc; i++) {
    if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      qf = i;
      break;
    }
  }
  if (qf == UINT32_MAX) {
    for (uint32_t i = 0; i < qfc; i++) {
      if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        qf = i;
        break;
      }
    }
  }
  free(qfs);

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf,
      .queueCount = 1,
      .pQueuePriorities = &prio,
  };
  VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
  };
  VkDevice dev;
  VK_CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, qf, 0, &q);

  VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qf,
  };
  VkCommandPool cpool;
  VK_CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cpool));
  VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cpool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

  VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK_CHECK(vkCreateFence(dev, &fci, NULL, &fence));

  VkPipelineCacheCreateInfo pcci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
  };
  VkPipelineCache cache;
  VK_CHECK(vkCreatePipelineCache(dev, &pcci, NULL, &cache));

  VkDeviceSize buf_sz = (VkDeviceSize)N_ELEMS * sizeof(float);
  float *h_a = (float *)malloc(buf_sz);
  float *h_b = (float *)malloc(buf_sz);
  for (uint32_t i = 0; i < N_ELEMS; i++) {
    h_a[i] = (float)(i & 1023);
    h_b[i] = (float)(i & 1023) * 0.5f;
  }

  VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer d_a, d_b, d_c, d_d, staging;
  VkDeviceMemory m_a, m_b, m_c, m_d, m_staging;
  make_buf(dev, phys, buf_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &d_a, &m_a);
  make_buf(dev, phys, buf_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &d_b, &m_b);
  make_buf(dev, phys, buf_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &d_c, &m_c);
  make_buf(dev, phys, buf_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &d_d, &m_d);
  make_buf(dev, phys, buf_sz,
           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
           &staging, &m_staging);

  void *mapped = NULL;
  VK_CHECK(vkMapMemory(dev, m_staging, 0, buf_sz, 0, &mapped));
  memcpy(mapped, h_a, buf_sz);
  vkUnmapMemory(dev, m_staging);
  copy_buf(dev, q, cmd, fence, staging, d_a, buf_sz);

  VK_CHECK(vkMapMemory(dev, m_staging, 0, buf_sz, 0, &mapped));
  memcpy(mapped, h_b, buf_sz);
  vkUnmapMemory(dev, m_staging);
  copy_buf(dev, q, cmd, fence, staging, d_b, buf_sz);

  run_shader_bench(dev, q, cmd, fence, cache, d_a, d_b, d_c, d_d, buf_sz,
                   staging, m_staging, h_a, h_b, "Regular GLSL (glslc -O)",
                   "bench_glsl.spv", "main");
  run_shader_bench(dev, q, cmd, fence, cache, d_a, d_b, d_c, d_d, buf_sz,
                   staging, m_staging, h_a, h_b, "MLIR GPU->SPIR-V",
                   "bench_mlir.spv", "saxpy");

  free(h_a);
  free(h_b);
  vkDestroyBuffer(dev, staging, NULL);
  vkFreeMemory(dev, m_staging, NULL);
  vkDestroyBuffer(dev, d_a, NULL);
  vkFreeMemory(dev, m_a, NULL);
  vkDestroyBuffer(dev, d_b, NULL);
  vkFreeMemory(dev, m_b, NULL);
  vkDestroyBuffer(dev, d_c, NULL);
  vkFreeMemory(dev, m_c, NULL);
  vkDestroyBuffer(dev, d_d, NULL);
  vkFreeMemory(dev, m_d, NULL);
  vkDestroyPipelineCache(dev, cache, NULL);
  vkDestroyFence(dev, fence, NULL);
  vkDestroyCommandPool(dev, cpool, NULL);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
  return 0;
}
