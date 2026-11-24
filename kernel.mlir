// MLIR GPU-dialect SAXPY compute kernel targeting Vulkan SPIR-V:
//   C[i] = 2.5 * A[i] + B[i]  for i in [0, 1024)
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    api=Vulkan,
    #spirv.resource_limits<
      max_compute_workgroup_invocations = 256,
      max_compute_workgroup_size = [256, 256, 64]
    >
  >
} {
  gpu.module @saxpy_module {
    gpu.func @saxpy(
      %A: memref<1024xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>},
      %B: memref<1024xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>},
      %C: memref<1024xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}
    ) kernel attributes {
      spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>
    } {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %base = arith.muli %bid, %bdim : index
      %idx = arith.addi %base, %tid : index

      %a = memref.load %A[%idx] : memref<1024xf32>
      %b = memref.load %B[%idx] : memref<1024xf32>
      %alpha = arith.constant 2.5 : f32
      %prod = arith.mulf %alpha, %a : f32
      %res = arith.addf %prod, %b : f32
      memref.store %res, %C[%idx] : memref<1024xf32>
      gpu.return
    }
  }
}
